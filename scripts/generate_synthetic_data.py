#!/usr/bin/env python3
"""Generate the synthetic CTAID POC dataset.

Every value written by this script is invented. Nothing is copied from a real
research table, a real scan, or a real person. Column names, dtypes and value
ranges come from ``app/schema.py``, which stores ranges only.

Usage::

    python scripts/generate_synthetic_data.py                # 700 records -> 2100 rows
    python scripts/generate_synthetic_data.py --records 900  # 2700 rows
    python scripts/generate_synthetic_data.py --seed 7

The default seed is fixed so the committed CSV can be reproduced byte for byte
with the same library versions (documented in docs/data-generation.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schema import (  # noqa: E402
    CLASS_LABELS,
    DERIVED_COLUMNS,
    EXCESS_KURTOSIS_OFFSET,
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    RECORD_ID_COLUMN,
    TARGET_COLUMN,
    VESSEL_COLUMN,
    VESSELS_PER_RECORD,
)

DEFAULT_SEED = 20261113
DEFAULT_RECORDS = 700
CLASS_PRIORS = (0.30, 0.26, 0.24, 0.20)
MIN_ROWS = VESSELS_PER_RECORD
MAX_ROWS = 30_000

# Each class shifts a handful of interpretable directions in the fake feature
# space so the demo models can learn something. Factors are tiny on purpose:
# this is a demo, not a claim about plaque appearance.
CLASS_EFFECTS: dict[int, dict[str, tuple[float, float]]] = {
    0: {
        "CONVENTIONAL_mean": (-0.55, 0.85),
        "CONVENTIONAL_max": (-0.45, 0.80),
        "CONVENTIONAL_calciumAgatstonScore[onlyForCT]": (-0.60, 0.70),
        "SHAPE_Volume(vx)": (-0.50, 0.90),
        "GLCM_Contrast[=Variance]": (-0.40, 0.80),
    },
    1: {
        "CONVENTIONAL_mean": (0.85, 0.90),
        "CONVENTIONAL_max": (0.75, 0.85),
        "CONVENTIONAL_calciumAgatstonScore[onlyForCT]": (1.10, 0.95),
        "SHAPE_Volume(vx)": (0.45, 1.00),
        "GLCM_Energy[=AngularSecondMoment]": (0.35, 0.90),
    },
    2: {
        "CONVENTIONAL_mean": (-0.30, 1.25),
        "CONVENTIONAL_std": (0.55, 1.10),
        "CONVENTIONAL_calciumAgatstonScore[onlyForCT]": (-0.85, 0.75),
        "GLCM_Homogeneity[=InverseDifference]": (0.50, 1.00),
        "SHAPE_Sphericity[onlyFor3DROI])": (-0.35, 1.05),
    },
    3: {
        "CONVENTIONAL_mean": (0.30, 1.10),
        "CONVENTIONAL_std": (0.35, 1.15),
        "CONVENTIONAL_calciumAgatstonScore[onlyForCT]": (0.55, 1.20),
        "GLCM_Contrast[=Variance]": (0.70, 1.10),
        "NGLDM_Contrast": (0.60, 1.05),
        "GLCM_Homogeneity[=InverseDifference]": (-0.45, 0.95),
    },
}

# Columns that must never go negative once sampled.
NON_NEGATIVE = {
    "CONVENTIONAL_max",
    "CONVENTIONAL_calciumAgatstonScore[onlyForCT]",
    "SHAPE_Volume(mL)",
    "SHAPE_Volume(vx)",
    "GLCM_Contrast[=Variance]",
    "GLRLM_HGRE",
    "GLRLM_SRHGE",
    "GLRLM_LRHGE",
    "NGLDM_Contrast",
    "GLZLM_HGZE",
    "GLZLM_SZHGE",
    "GLZLM_LZHGE",
}

def _sample_column(rng: np.random.Generator, spec: dict, class_ids: np.ndarray) -> np.ndarray:
    """Draw one column for every row, with a small per-class shift."""
    name = str(spec["name"])
    lo = float(spec["min"])
    hi = float(spec["max"])
    mean = float(spec["mean"])
    std = float(spec["std"])

    # Most of these features are strictly positive with a long right tail, so a
    # log normal draw keeps the shape plausible. Signed features use a plain
    # normal draw around the recorded mean.
    if lo >= 0 and mean > 0 and (hi / max(mean, 1e-9)) < 12:
        sigma = float(np.clip(std / max(mean, 1e-9), 0.02, 0.9))
        z = rng.normal(0.0, sigma, size=class_ids.shape[0])
        base = mean * np.exp(z - 0.5 * sigma * sigma)
    else:
        sigma = max(std, 1e-6)
        base = mean + rng.normal(0.0, sigma, size=class_ids.shape[0])

    # Per-row class effect: a shift in standard deviation units plus a scale.
    shift = np.zeros(class_ids.shape[0])
    scale = np.ones(class_ids.shape[0])
    for class_id, effects in CLASS_EFFECTS.items():
        effect = effects.get(name)
        if effect is None:
            continue
        mask = class_ids == class_id
        if not mask.any():
            continue
        shift[mask] = effect[0]
        scale[mask] = effect[1]

    base = (base + shift * std) * scale

    values = np.clip(base, lo, hi)
    if spec["dtype"] == "int":
        values = np.rint(values)
    return values


def build_frame(records: int = DEFAULT_RECORDS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return the full synthetic table with Record ID, Vessel and Diagnosis."""
    if records < 1:
        raise ValueError("records must be at least 1")
    rows = records * VESSELS_PER_RECORD
    if rows > MAX_ROWS:
        raise ValueError(f"refusing to generate more than {MAX_ROWS} rows")

    rng = np.random.default_rng(seed)

    record_ids = np.repeat(np.arange(1, records + 1), VESSELS_PER_RECORD)
    vessel_numbers = np.tile(np.arange(1, VESSELS_PER_RECORD + 1), records)

    # One latent class per record, so all three vessels of a record agree.
    record_class = rng.choice(len(CLASS_PRIORS), size=records, p=list(CLASS_PRIORS))
    diagnosis = np.repeat(record_class, VESSELS_PER_RECORD)

    data: dict[str, np.ndarray] = {
        RECORD_ID_COLUMN: record_ids,
        VESSEL_COLUMN: vessel_numbers,
    }

    for spec in FEATURE_COLUMNS:
        name = str(spec["name"])
        data[name] = _sample_column(rng, spec, diagnosis)
        # Vessels inside one record stay correlated: nudge each vessel a little.
        if VESSELS_PER_RECORD > 1:
            drift = rng.normal(0.0, 0.04, size=records)
            data[name] = data[name] + np.repeat(drift, VESSELS_PER_RECORD) * float(spec["std"])
            lo, hi = float(spec["min"]), float(spec["max"])
            data[name] = np.clip(data[name], lo, hi)
            if spec["dtype"] == "int":
                data[name] = np.rint(data[name])
        if name in NON_NEGATIVE:
            data[name] = np.maximum(data[name], 0.0)

    for child, parent in DERIVED_COLUMNS.items():
        data[child] = data[parent] - EXCESS_KURTOSIS_OFFSET

    data[TARGET_COLUMN] = diagnosis

    frame = pd.DataFrame(data)
    frame = frame[[RECORD_ID_COLUMN, VESSEL_COLUMN, *FEATURE_NAMES, TARGET_COLUMN]]
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "synthetic_features.csv",
        help="output CSV path (default: data/synthetic_features.csv)",
    )
    args = parser.parse_args(argv)

    frame = build_frame(records=args.records, seed=args.seed)
    write_csv(args.out, frame)

    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    print(f"seed={args.seed} records={args.records} rows={len(frame)} cols={frame.shape[1]}")
    print(f"wrote {args.out}")
    print("class counts: " + ", ".join(f"{CLASS_LABELS[int(k)]}={int(v)}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
