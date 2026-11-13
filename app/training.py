"""Model training for the CTAID POC demo.

Training really runs: scikit-learn fits on the synthetic CSV and the fitted
pipeline is written to ``models/`` as a joblib artifact. Metrics reported in the
UI come from the held out split, never from hard coded numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.data import feature_columns
from app.schema import TARGET_COLUMN

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "family": "linear",
        "note": "Fast, well calibrated baseline with standard scaling.",
    },
    "random_forest": {
        "label": "Random Forest",
        "family": "trees",
        "note": "Non linear, handles feature interactions, gives importances.",
    },
    "svm": {
        "label": "Support Vector Machine",
        "family": "kernel",
        "note": "RBF kernel with probability estimates, strong on scaled features.",
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "family": "boosting",
        "note": "Additive trees, usually the best accuracy on tabular data.",
    },
}

DEFAULT_MODELS = ("logistic_regression", "random_forest")


@dataclass(frozen=True)
class TrainConfig:
    models: tuple[str, ...] = DEFAULT_MODELS
    test_size: float = 0.25
    random_state: int = 42
    sample_limit: int | None = None
    artifact_dir: Path | None = None
    persist: bool = True

    def validate(self) -> None:
        if not self.models:
            raise ValueError("Select at least one model")
        unknown = [name for name in self.models if name not in MODEL_SPECS]
        if unknown:
            raise ValueError("Unknown model(s): " + ", ".join(unknown))
        if not 0.1 <= self.test_size <= 0.6:
            raise ValueError("test_size must be between 0.1 and 0.6")


@dataclass
class ModelResult:
    key: str
    label: str
    metrics: dict[str, float]
    report: dict[str, Any]
    confusion: np.ndarray
    labels: list[int]
    roc: dict[int, dict[str, list[float] | float]]
    importance: list[tuple[str, float]]
    train_seconds: float
    artifact: str | None = None
    pipeline: Any = field(default=None, repr=False)


@dataclass
class TrainingRun:
    config: TrainConfig
    feature_names: list[str]
    labels: list[int]
    results: list[ModelResult]
    rows_used: int
    rows_train: int
    rows_test: int
    class_counts: dict[int, int]
    finished_at: str = ""

    def best(self) -> ModelResult:
        return max(self.results, key=lambda result: result.metrics["roc_auc_macro"])

    def to_record(self) -> dict[str, Any]:
        """Serialisable summary, safe for logs, Convex and the sidebar."""
        return {
            "finished_at": self.finished_at,
            "rows_used": self.rows_used,
            "rows_train": self.rows_train,
            "rows_test": self.rows_test,
            "feature_count": len(self.feature_names),
            "class_counts": self.class_counts,
            "models": [
                {
                    "key": result.key,
                    "label": result.label,
                    "metrics": result.metrics,
                    "artifact": result.artifact,
                    "train_seconds": round(result.train_seconds, 3),
                }
                for result in self.results
            ],
        }


def build_pipeline(key: str, random_state: int) -> Pipeline:
    """Return an unfitted pipeline for one registry key."""
    if key == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    elif key == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    elif key == "svm":
        # Calibrated wrapper keeps predict_proba available without the
        # deprecated SVC(probability=True) flag.
        estimator = CalibratedClassifierCV(
            SVC(kernel="rbf", C=2.0, gamma="scale", random_state=random_state),
            method="sigmoid",
            cv=3,
        )
    elif key == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=random_state)
    else:  # pragma: no cover - guarded by TrainConfig.validate
        raise ValueError(f"Unknown model: {key}")

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def _importance(pipeline: Pipeline, feature_names: list[str]) -> list[tuple[str, float]]:
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:  # pragma: no cover - SVC without linear kernel
        return []
    if values.shape[0] != len(feature_names):
        return []
    pairs = sorted(zip(feature_names, values, strict=True), key=lambda item: abs(item[1]), reverse=True)
    return [(name, float(value)) for name, value in pairs[:12]]


def _roc_curves(estimator: Pipeline, X_test: np.ndarray, y_test: np.ndarray, labels: list[int]):
    curves: dict[int, dict[str, list[float] | float]] = {}
    try:
        proba = estimator.predict_proba(X_test)
    except Exception:  # pragma: no cover - estimators without predict_proba
        return curves
    classes = list(estimator.classes_)
    for label in labels:
        if label not in classes:
            continue
        column = classes.index(label)
        one_hot = (y_test == label).astype(int)
        if one_hot.sum() == 0 or one_hot.sum() == len(one_hot):
            continue
        fpr, tpr, _ = roc_curve(one_hot, proba[:, column])
        curves[label] = {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
            "auc": round(float(roc_auc_score(one_hot, proba[:, column])), 4),
        }
    return curves


def train_models(frame: pd.DataFrame, config: TrainConfig) -> TrainingRun:
    """Fit every requested model on the synthetic data and collect metrics."""
    config.validate()
    features = feature_columns(frame)
    data = frame
    if config.sample_limit is not None and len(data) > config.sample_limit:
        data = data.sample(n=config.sample_limit, random_state=config.random_state)

    X = data[features].to_numpy(dtype=float)
    y = data[TARGET_COLUMN].to_numpy(dtype=int)

    class_counts = {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}
    stratify = y if min(class_counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    labels = sorted(int(v) for v in set(y.tolist()))
    results: list[ModelResult] = []

    for key in config.models:
        started = time.perf_counter()
        pipeline = build_pipeline(key, config.random_state)
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - started

        y_pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)
        classes = list(pipeline.classes_)

        try:
            auc_macro = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=classes))
        except ValueError:  # pragma: no cover - single class fold
            auc_macro = float("nan")

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "roc_auc_macro": round(auc_macro, 4),
        }

        report = classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        artifact = None
        if config.persist and config.artifact_dir is not None:
            artifact = str(save_artifact(pipeline, config.artifact_dir, key, metrics))

        results.append(
            ModelResult(
                key=key,
                label=MODEL_SPECS[key]["label"],
                metrics=metrics,
                report=report,
                confusion=confusion_matrix(y_test, y_pred, labels=labels),
                labels=labels,
                roc=_roc_curves(pipeline, X_test, y_test, labels),
                importance=_importance(pipeline, features),
                train_seconds=elapsed,
                artifact=artifact,
                pipeline=pipeline,
            )
        )

    return TrainingRun(
        config=config,
        feature_names=features,
        labels=labels,
        results=results,
        rows_used=int(len(data)),
        rows_train=int(len(X_train)),
        rows_test=int(len(X_test)),
        class_counts=class_counts,
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def save_artifact(pipeline: Pipeline, directory: Path, key: str, metrics: dict[str, float]) -> Path:
    """Persist the fitted pipeline plus a sidecar metric file."""
    import joblib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{key}-{stamp}.joblib"
    joblib.dump(pipeline, path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "model": key,
                "trained_at": stamp,
                "metrics": metrics,
                "data": "synthetic_features.csv",
                "disclaimer": "Demo with synthetic data, not a medical tool.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    latest = directory / f"{key}-latest.joblib"
    joblib.dump(pipeline, latest)
    latest.with_suffix(".json").write_text(sidecar.read_text(encoding="utf-8"), encoding="utf-8")
    return latest


def load_artifact(directory: Path, key: str) -> Pipeline:
    import joblib

    path = Path(directory) / f"{key}-latest.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"No trained artifact for {key} at {path}")
    return joblib.load(path)


def ranked_features(frame: pd.DataFrame, features: list[str] | None = None, limit: int = 6) -> list[tuple[str, float]]:
    """Cheap univariate separability ranking, used for charts and hints.

    Uses an F score between feature and class label, so it needs no fitted
    model and stays fast on the full synthetic table.
    """
    from sklearn.feature_selection import f_classif

    names = features or feature_columns(frame)
    X = frame[names].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores, _ = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
    pairs = sorted(zip(names, scores.tolist(), strict=True), key=lambda item: item[1], reverse=True)
    return [(name, round(float(score), 3)) for name, score in pairs[:limit]]
