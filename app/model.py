"""Load the bundled demo model and run a prediction."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from app.features import FEATURE_NAMES, FEATURE_SPECS
from app.synthetic import MODEL_PATH, train_and_save


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    import joblib

    if not MODEL_PATH.is_file():
        train_and_save(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def metrics() -> dict:
    return dict(load_bundle()["metrics"])


def _validate(payload: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for spec in FEATURE_SPECS:
        name = spec["name"]
        if name not in payload:
            raise ValueError(f"Missing field: {name}")
        try:
            raw = float(payload[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {name}") from exc
        if spec["kind"] == "binary":
            if raw not in (0.0, 1.0):
                raise ValueError(f"{name} must be 0 or 1")
        else:
            lo, hi = float(spec["min"]), float(spec["max"])
            if raw < lo or raw > hi:
                raise ValueError(f"{name} must be between {lo} and {hi}")
        values.append(raw)
    return values


def predict(payload: dict[str, Any]) -> dict:
    values = _validate(payload)
    bundle = load_bundle()
    model = bundle["model"]
    X = np.array([values], dtype=float)
    proba = float(model.predict_proba(X)[0, 1])
    label = "higher synthetic risk flag" if proba >= 0.5 else "lower synthetic risk flag"
    return {
        "probability": round(proba, 3),
        "label": label,
        "features_used": FEATURE_NAMES,
        "disclaimer": (
            "Demo with synthetic data, not a medical tool. "
            "This score is produced by a small model trained only on invented data. "
            "It is not a diagnosis and must not be used for clinical decisions."
        ),
    }
