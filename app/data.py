"""Dataset access for the CTAID POC demo.

The only dataset this app ships is ``data/synthetic_features.csv``, produced by
``scripts/generate_synthetic_data.py``. Nothing here reads a real research file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.schema import (
    CLASS_DESCRIPTIONS,
    CLASS_LABELS,
    RECORD_ID_COLUMN,
    TARGET_COLUMN,
    VESSEL_COLUMN,
)


class DatasetError(RuntimeError):
    """Raised when the synthetic dataset is missing or malformed."""


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    records: int
    feature_count: int
    class_counts: dict[int, int]
    missing_cells: int

    @property
    def class_balance(self) -> dict[int, float]:
        total = sum(self.class_counts.values()) or 1
        return {key: value / total for key, value in self.class_counts.items()}


def resolve_dataset_path(path: Path) -> Path:
    """Return an existing dataset path, generating the demo data if absent."""
    if path.is_file():
        return path
    from scripts.generate_synthetic_data import build_frame  # local import, optional

    frame = build_frame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the synthetic table and validate its shape."""
    path = resolve_dataset_path(Path(path))
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - corrupted file path
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc

    missing = {RECORD_ID_COLUMN, VESSEL_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise DatasetError("Dataset is missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise DatasetError("Dataset is empty")

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric modelling columns: everything except id, vessel and target."""
    drop = {RECORD_ID_COLUMN, VESSEL_COLUMN, TARGET_COLUMN}
    return [col for col in frame.columns if col not in drop]


def summary(frame: pd.DataFrame) -> DatasetSummary:
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    return DatasetSummary(
        rows=int(len(frame)),
        columns=int(frame.shape[1]),
        records=int(frame[RECORD_ID_COLUMN].nunique()),
        feature_count=len(feature_columns(frame)),
        class_counts={int(k): int(v) for k, v in counts.items()},
        missing_cells=int(frame.isna().sum().sum()),
    )


def describe(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics in a stable, display friendly order."""
    numeric = frame[feature_columns(frame)]
    stats = numeric.describe().T
    stats = stats[["count", "mean", "std", "min", "50%", "max"]]
    stats.columns = ["count", "mean", "std", "min", "median", "max"]
    stats.insert(0, "feature", stats.index)
    return stats.reset_index(drop=True)


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    numeric = frame[feature_columns(frame)]
    return numeric.corr(method=method, numeric_only=True)


def top_correlations(matrix: pd.DataFrame, limit: int = 12) -> list[tuple[str, str, float]]:
    """Strongest absolute feature pairs, with self pairs removed."""
    pairs: list[tuple[str, str, float]] = []
    cols = list(matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = matrix.loc[left, right]
            if pd.isna(value):
                continue
            pairs.append((left, right, float(value)))
    pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    return pairs[:limit]


def class_label(class_id: int) -> str:
    return CLASS_LABELS.get(int(class_id), f"Class {int(class_id)}")


def class_caption(class_id: int) -> str:
    return CLASS_DESCRIPTIONS.get(int(class_id), "synthetic pattern group")


def sample_for_class(frame: pd.DataFrame, class_id: int, seed: int | None = None) -> pd.DataFrame:
    """One synthetic row from a chosen class, for the prediction page.

    These are rows of the generated demo data, not original sample files.
    """
    subset = frame[frame[TARGET_COLUMN] == int(class_id)]
    if subset.empty:
        raise DatasetError(f"No synthetic rows for class {class_id}")
    rng = np.random.default_rng(seed)
    index = int(rng.integers(0, len(subset)))
    return subset.iloc[[index]].reset_index(drop=True)


def class_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Per class mean of the most separable features, for the Home page."""
    from app.training import ranked_features

    features = feature_columns(frame)
    top = [name for name, _ in ranked_features(frame, features, limit=6)]
    grouping = frame.groupby(TARGET_COLUMN)[top].mean().T
    grouping.columns = [class_label(int(col)) for col in grouping.columns]
    grouping.index.name = "feature"
    return grouping.reset_index()
