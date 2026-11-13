"""Generate fully synthetic tabular data and train a small demo classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features import FEATURE_NAMES

SEED = 42
N_ROWS = 2500
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "demo_model.joblib"


def generate_dataset(n: int = N_ROWS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Invent a labelled table. No real clinical records are used."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 91, n)
    sbp = rng.normal(128, 18, n).clip(90, 200)
    chol = rng.normal(205, 38, n).clip(120, 350)
    calcium = rng.gamma(1.6, 45, n).clip(0, 800)
    stenosis = (rng.beta(2.0, 5.0, n) * 85).clip(0, 90)
    smoking = rng.integers(0, 2, n)
    diabetes = rng.integers(0, 2, n)

    logit = (
        -4.2
        + 0.045 * (age - 50)
        + 0.016 * (sbp - 120)
        + 0.008 * (chol - 200)
        + 0.006 * calcium
        + 0.045 * stenosis
        + 0.7 * smoking
        + 0.8 * diabetes
        + rng.normal(0, 0.55, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    X = np.column_stack([age, sbp, chol, calcium, stenosis, smoking, diabetes])
    return X, y


def train_and_save(path: Path = MODEL_PATH) -> dict:
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=500, random_state=SEED),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "feature_names": FEATURE_NAMES,
        "source": "synthetic-generator-only",
        "seed": SEED,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "metrics": metrics}, path)
    return metrics
