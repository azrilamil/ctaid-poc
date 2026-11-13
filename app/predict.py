"""Prediction path and upload validation for the CTAID POC demo."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import MAX_UPLOAD_BYTES, MAX_UPLOAD_ROWS, REQUIRED_UPLOAD_SUFFIX
from app.privacy import IDENTIFIER_COLUMN
from app.schema import CLASS_LABELS, DISPLAY_LABELS, RECORD_ID_COLUMN, TARGET_COLUMN, TARGET_DISPLAY, VESSEL_COLUMN

DROP_ON_UPLOAD = {RECORD_ID_COLUMN, IDENTIFIER_COLUMN, VESSEL_COLUMN, TARGET_COLUMN, TARGET_DISPLAY}


class UploadError(ValueError):
    """Raised when an uploaded CSV fails validation."""


@dataclass(frozen=True)
class UploadReport:
    name: str
    size_bytes: int
    rows: int
    columns: int
    dropped_columns: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "columns": self.columns,
            "dropped_columns": list(self.dropped_columns),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class Prediction:
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: dict[int, float]
    top_features: list[tuple[str, float, float]]
    model_key: str
    rows_scored: int


def validate_upload(filename: str | None, payload: bytes, content_type: str | None = None) -> bytes:
    """Reject anything that is not a small CSV. Returns the payload unchanged."""
    if not filename:
        raise UploadError("Choose a CSV file to upload.")
    if not filename.lower().endswith(REQUIRED_UPLOAD_SUFFIX):
        raise UploadError("Only .csv files are accepted.")
    if not payload:
        raise UploadError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadError(
            f"File is {len(payload) / 1024:.0f} KB, over the {MAX_UPLOAD_BYTES // 1024} KB demo limit."
        )
    if content_type:
        allowed = ("text/csv", "text/plain", "application/csv", "application/vnd.ms-excel", "application/octet-stream")
        if not any(content_type.startswith(value) for value in allowed):
            raise UploadError(f"Unexpected content type: {content_type}")
    head = payload[:4096].decode("utf-8", errors="ignore")
    if "\x00" in head or "\ufffd" in head:
        raise UploadError("File does not look like text. Binary uploads are rejected.")
    return payload


def read_upload(filename: str | None, payload: bytes, content_type: str | None = None) -> tuple[pd.DataFrame, UploadReport]:
    """Validate then parse an uploaded CSV into a frame."""
    validate_upload(filename, payload, content_type)
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise UploadError(f"Could not parse CSV: {exc}") from exc

    if frame.empty:
        raise UploadError("The CSV has no rows.")
    if len(frame) > MAX_UPLOAD_ROWS:
        raise UploadError(f"CSV has {len(frame)} rows, over the {MAX_UPLOAD_ROWS} row demo limit.")

    warnings: list[str] = []
    dropped = tuple(col for col in frame.columns if col in DROP_ON_UPLOAD)
    if dropped:
        frame = frame.drop(columns=list(dropped))
        shown = ", ".join(DISPLAY_LABELS.get(name, name) for name in dropped)
        warnings.append("Dropped non modelling column(s): " + shown + ".")

    report = UploadReport(
        name=filename or "upload.csv",
        size_bytes=len(payload),
        rows=int(len(frame)),
        columns=int(frame.shape[1]),
        dropped_columns=dropped,
        warnings=tuple(warnings),
    )
    return frame, report


def align_features(frame: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Reorder and check the feature frame against the trained feature list."""
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise UploadError(f"Missing {len(missing)} required feature column(s): {preview}{more}.")
    extra = [name for name in frame.columns if name not in feature_names]
    aligned = frame[feature_names].astype(float)
    return aligned, extra


def feature_weights(pipeline, feature_names: list[str]) -> dict[str, float]:
    """Real per feature weights from the fitted model.

    Tree ensembles expose ``feature_importances_``. Linear models expose
    ``coef_``, so the absolute mean coefficient across classes is used. A
    calibrated wrapper is unwrapped to its fitted base estimator. When nothing
    is available the mapping stays empty and the UI just omits the table.
    """
    estimator = None
    if hasattr(pipeline, "named_steps"):
        estimator = pipeline.named_steps.get("model")
    if estimator is None:
        return {}

    # Unwrap calibration wrappers and voting style ensembles.
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated:
        fitted = getattr(calibrated[0], "estimator", None) or getattr(calibrated[0], "base_estimator", None)
        if fitted is not None:
            estimator = fitted

    values: np.ndarray | None = None
    raw = getattr(estimator, "feature_importances_", None)
    if raw is not None:
        values = np.asarray(raw, dtype=float)
    else:
        coef = getattr(estimator, "coef_", None)
        if coef is not None:
            coef = np.asarray(coef, dtype=float)
            values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or values.shape[0] != len(feature_names):
        return {}
    return {name: float(weight) for name, weight in zip(feature_names, values, strict=True)}


def predict_frame(pipeline, frame: pd.DataFrame, feature_names: list[str], model_key: str = "model") -> list[Prediction]:
    """Score every row of a curated frame."""
    aligned, _ = align_features(frame, feature_names)
    values = aligned.to_numpy(dtype=float)
    classes = [int(c) for c in pipeline.classes_]
    proba = pipeline.predict_proba(values)
    predicted = pipeline.predict(values)

    importances = feature_weights(pipeline, feature_names)
    out: list[Prediction] = []
    for row_index in range(len(aligned)):
        row_proba = {cls: round(float(proba[row_index][pos]), 4) for pos, cls in enumerate(classes)}
        weighted = [
            (name, float(values[row_index][pos]), float(importances[name]))
            for pos, name in enumerate(feature_names)
            if importances.get(name, 0.0) > 0.0
        ]
        weighted.sort(key=lambda item: item[2], reverse=True)
        top = weighted[:5]
        cls = int(predicted[row_index])
        out.append(
            Prediction(
                predicted_class=cls,
                predicted_label=CLASS_LABELS.get(cls, f"Class {cls}"),
                confidence=row_proba.get(cls, max(row_proba.values()) if row_proba else 0.0),
                probabilities=row_proba,
                top_features=top,
                model_key=model_key,
                rows_scored=len(aligned),
            )
        )
    return out


def prediction_record(prediction: Prediction, source: str = "form") -> dict[str, object]:
    """Shape a prediction for the history table (synthetic demo traffic only)."""
    return {
        "source": source,
        "predicted_class": prediction.predicted_class,
        "predicted_label": prediction.predicted_label,
        "confidence": prediction.confidence,
        "probabilities": {str(k): v for k, v in prediction.probabilities.items()},
        "top_features": [name for name, _value, _weight in prediction.top_features],
    }


def load_sample_csv(path: Path) -> pd.DataFrame:
    """Read one generated sample CSV (never an original sample_data file)."""
    return pd.read_csv(path)
