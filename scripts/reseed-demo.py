#!/usr/bin/env python3
"""Reset the CTAID POC demo to a clean state and reseed the synthetic data.

What it does:

1. Regenerates ``data/synthetic_features.csv`` from the fixed seed.
2. Clears prediction history and model version rows (SQLite and/or Convex).
3. Clears uploaded CSVs and model artifacts from object storage
   (Garage S3 when configured, local ``var/objects`` otherwise).
4. Re-registers the dataset metadata.

Schedule it daily, for example with cron::

    15 3 * * *  cd /srv/ctaid-poc && python scripts/reseed-demo.py >> var/reseed.log 2>&1

The script never touches anything outside ``data/``, ``models/`` and ``var/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backend import build_backend  # noqa: E402
from app.bootstrap import ensure_dataset_meta  # noqa: E402
from app.config import Settings  # noqa: E402
from app.storage import build_store  # noqa: E402

KEEP_SUFFIXES = {".README.md", ".gitkeep"}


def clear_models(directory: Path, dry_run: bool) -> list[str]:
    removed: list[str] = []
    if not directory.is_dir():
        return removed
    for path in sorted(directory.glob("*")):
        if path.name.lower() == "readme.md" or path.name in {".gitkeep"}:
            continue
        if path.is_file():
            removed.append(str(path.relative_to(ROOT)))
            if not dry_run:
                path.unlink()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report actions without changing anything")
    parser.add_argument("--keep-models", action="store_true", help="Keep trained joblib artifacts")
    parser.add_argument("--records", type=int, default=None, help="Override record count for the reseed")
    args = parser.parse_args(argv)

    settings = Settings.load()
    store = build_store(settings.storage, settings.state_dir / "objects")
    backend = build_backend(settings.convex, settings.state_dir / "ctaid-demo.sqlite3")

    print(f"backend: {backend.name} | storage: {store.backend}")

    if args.dry_run:
        print(f"would regenerate {settings.dataset_path}")
        print("would reset prediction and model history")
        print(f"would clear uploads under uploads/ ({len(store.list_keys('uploads/'))} objects)")
        if not args.keep_models:
            print("would remove model artifacts")
        return 0

    # 1. regenerate the synthetic CSV
    from scripts.generate_synthetic_data import build_frame, write_csv

    frame = build_frame(records=args.records) if args.records else build_frame()
    write_csv(settings.dataset_path, frame)
    print(f"regenerated {settings.dataset_path} rows={len(frame)}")

    # 2. clear history
    backend.reset()
    print("cleared prediction history and model versions")

    # 3. clear uploads
    removed_objects = store.clear()
    print(f"cleared {removed_objects} stored object(s)")

    # 4. clear model artifacts
    if args.keep_models:
        print("kept model artifacts")
    else:
        removed = clear_models(settings.models_dir, dry_run=False)
        print(f"removed {len(removed)} model artifact(s)")

    # 5. re-register metadata
    meta = ensure_dataset_meta(settings)
    print(f"registered dataset meta rows={meta['rows']} digest={meta['digest']}")
    print("Demo with synthetic data, not a medical tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
