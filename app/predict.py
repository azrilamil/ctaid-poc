"""Prediction path and upload validation for the CTAID POC demo."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import MAX_UPLOAD_BYTES, MAX_UPLOAD_ROWS, REQUIRED_UPLOAD_SUFFIX
from app.schema import CLASS_LABELS, RECORD_ID_COLUMN, TARGET_COLUMN, VESSEL_COLUMN

DROP_ON_UPLOAD = {RECORD_ID_COLUMN, "Patient ID", VESSEL_COLUMN, TARGET_COLUMN}


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
        warnings.append("Dropped non modelling column(s): " + ", ".join(dropped) + ".")

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


def predict_frame(pipeline, frame: pd.DataFrame, feature_names: list[str], model_key: str = "model") -> list[Prediction]:
    """Score every row of a curated frame."""
    aligned, _ = align_features(frame, feature_names)
    values = aligned.to_numpy(dtype=float)
    classes = [int(c) for c in pipeline.classes_]
    proba = pipeline.predict_proba(values)
    predicted = pipeline.predict(values)

    importances = {}
    estimator = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else None
    if estimator is not None and hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_, dtype=float)
        if len(raw) == len(feature_names):
            importances = dict(zip(feature_names, raw, strict=True))

    out: list[Prediction] = []
    for row_index in range(len(aligned)):
        row_proba = {cls: round(float(proba[row_index][pos]), 4) for pos, cls in enumerate(classes)}
        top = sorted(
            (
                (name, float(values[row_index][pos]), float(importances.get(name, 0.0)))
                for pos, name in enumerate(feature_names)
            ),
            key=lambda item: item[2],
            reverse=True,
        )[:5]
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
