#!/usr/bin/env python3
"""Train the bundled demo model from synthetic data only."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.synthetic import MODEL_PATH, train_and_save  # noqa: E402

if __name__ == "__main__":
    metrics = train_and_save(MODEL_PATH)
    print(json.dumps({"model_path": str(MODEL_PATH), **metrics}, indent=2))
