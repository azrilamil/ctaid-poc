"""One-time demo bootstrap: register dataset metadata with the backend.

Runs at app start, writes the synthetic CSV if it is missing, and records a
digest so the sidebar and docs can quote the exact dataset in use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.backend import build_backend
from app.config import Settings
from app.data import load_dataset, resolve_dataset_path, summary


def dataset_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def ensure_dataset_meta(settings: Settings) -> dict:
    """Make sure the CSV exists and the backend knows about it."""
    path = resolve_dataset_path(settings.dataset_path)
    frame = load_dataset(path)
    info = summary(frame)
    record = {
        "name": path.name,
        "rows": info.rows,
        "columns": info.columns,
        "records": info.records,
        "class_counts": {str(k): v for k, v in info.class_counts.items()},
        "digest": dataset_digest(path),
    }
    backend = build_backend(settings.convex, settings.state_dir / "ctaid-demo.sqlite3")
    try:
        saved = backend.save_dataset_meta(record)
    except Exception:  # pragma: no cover - backend guard
        saved = record
    return {**record, "backend": saved.get("backend", backend.name)}


def main() -> int:
    settings = Settings.load()
    info = ensure_dataset_meta(settings)
    print(
        f"dataset {info['name']} rows={info['rows']} columns={info['columns']} "
        f"records={info['records']} digest={info['digest']} backend={info['backend']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
