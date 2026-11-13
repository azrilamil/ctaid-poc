#!/usr/bin/env python3
"""Train the demo artifacts used by the Prediction page.

Writes ``models/<model>-latest.joblib`` for the requested models plus a JSON
sidecar next to each one. Runs in a few seconds on the synthetic table.

Usage::

    python scripts/train_demo_model.py
    python scripts/train_demo_model.py --models logistic_regression random_forest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_DATASET, MODELS_DIR  # noqa: E402
from app.data import load_dataset  # noqa: E402
from app.training import DEFAULT_MODELS, MODEL_SPECS, TrainConfig, train_models  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(MODEL_SPECS))
    parser.add_argument("--rows", type=int, default=None, help="Optional row cap for a fast run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)

    frame = load_dataset(args.dataset)
    config = TrainConfig(
        models=tuple(args.models),
        random_state=args.seed,
        sample_limit=args.rows,
        artifact_dir=args.out,
        persist=True,
    )
    run = train_models(frame, config)

    print(f"rows used {run.rows_used} (train {run.rows_train} / test {run.rows_test})")
    for result in run.results:
        metrics = " ".join(f"{key}={value:.4f}" for key, value in result.metrics.items())
        print(f"{result.label}: {metrics}")
        print(f"  artifact {result.artifact}")
    print("Demo with synthetic data, not a medical tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
