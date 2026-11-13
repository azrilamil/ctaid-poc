"""Persistence for demo metadata: Convex when configured, SQLite otherwise.

Three collections are kept: dataset metadata, model versions, prediction
history. Everything stored here is synthetic demo traffic. No accounts, no real
records, no personal data.

The Convex path talks to the HTTP API directly (``/api/query`` and
``/api/mutation``) so Streamlit needs no JS bridge. When ``CONVEX_URL`` is unset
the SQLite backend in ``var/ctaid-demo.sqlite3`` is used and behaves the same.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import ConvexConfig

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dataset_meta (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    rows INTEGER NOT NULL,
    columns INTEGER NOT NULL,
    records INTEGER NOT NULL,
    class_counts TEXT NOT NULL,
    digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    label TEXT NOT NULL,
    metrics TEXT NOT NULL,
    rows_used INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    artifact TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    predicted_class INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL,
    top_features TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SQLiteBackend:
    name = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success and always close it."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # dataset metadata ---------------------------------------------------
    def save_dataset_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": meta.get("id") or uuid.uuid4().hex[:12],
            "name": meta["name"],
            "rows": int(meta["rows"]),
            "columns": int(meta["columns"]),
            "records": int(meta["records"]),
            "class_counts": json.dumps(meta.get("class_counts", {})),
            "digest": str(meta.get("digest", "")),
            "created_at": meta.get("created_at") or _now(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO dataset_meta (id, name, rows, columns, records, class_counts, digest, created_at)"
                " VALUES (:id, :name, :rows, :columns, :records, :class_counts, :digest, :created_at)",
                row,
            )
        return row

    def latest_dataset_meta(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            found = conn.execute("SELECT * FROM dataset_meta ORDER BY created_at DESC LIMIT 1").fetchone()
        return dict(found) if found else None

    # model versions -----------------------------------------------------
    def save_model_version(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": record.get("id") or uuid.uuid4().hex[:12],
            "model": record["model"],
            "label": record.get("label", record["model"]),
            "metrics": json.dumps(record.get("metrics", {})),
            "rows_used": int(record.get("rows_used", 0)),
            "feature_count": int(record.get("feature_count", 0)),
            "artifact": record.get("artifact"),
            "created_at": record.get("created_at") or _now(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO model_versions"
                " (id, model, label, metrics, rows_used, feature_count, artifact, created_at)"
                " VALUES (:id, :model, :label, :metrics, :rows_used, :feature_count, :artifact, :created_at)",
                row,
            )
        return row

    def list_model_versions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["metrics"] = json.loads(item["metrics"])
            out.append(item)
        return out

    # predictions --------------------------------------------------------
    def save_prediction(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": record.get("id") or uuid.uuid4().hex[:12],
            "source": record.get("source", "form"),
            "predicted_class": int(record["predicted_class"]),
            "predicted_label": record["predicted_label"],
            "confidence": float(record["confidence"]),
            "probabilities": json.dumps(record.get("probabilities", {})),
            "top_features": json.dumps(record.get("top_features", [])),
            "created_at": record.get("created_at") or _now(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions"
                " (id, source, predicted_class, predicted_label, confidence, probabilities, top_features, created_at)"
                " VALUES (:id, :source, :predicted_class, :predicted_label, :confidence, :probabilities,"
                " :top_features, :created_at)",
                row,
            )
        return row

    def list_predictions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["probabilities"] = json.loads(item["probabilities"])
            item["top_features"] = json.loads(item["top_features"])
            out.append(item)
        return out

    def count_predictions(self) -> int:
        with self._connect() as conn:
            found = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()
        return int(found["n"]) if found else 0

    def reset(self) -> None:
        with self._connect() as conn:
            conn.executescript("DELETE FROM predictions; DELETE FROM model_versions;")


class ConvexBackend:
    """Thin HTTP client for Convex functions.

    Falls back to SQLite per call if the deployment is unreachable, so a dead
    URL never breaks the demo. ``name`` reports which path served the call.
    """

    name = "convex"

    def __init__(self, config: ConvexConfig, fallback: SQLiteBackend) -> None:
        self.config = config
        self.fallback = fallback

    def _call(self, kind: str, path: str, args: dict[str, Any]) -> Any:
        import requests

        url = f"{self.config.url.rstrip('/')}/api/{kind}"
        headers = {"Content-Type": "application/json"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        payload = {"path": path, "args": {**args, "format": "json"}, "format": "json"}
        response = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout_s)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(body.get("errorMessage", "Convex call failed"))
        return body.get("value")

    def save_dataset_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        try:
            self._call("mutation", "ctaid:saveDatasetMeta", {"meta": meta})
            return dict(meta, backend=self.name)
        except Exception:
            return dict(self.fallback.save_dataset_meta(meta), backend="sqlite-fallback")

    def latest_dataset_meta(self) -> dict[str, Any] | None:
        try:
            value = self._call("query", "ctaid:latestDatasetMeta", {})
            if isinstance(value, dict):
                return dict(value, backend=self.name)
        except Exception:
            pass
        found = self.fallback.latest_dataset_meta()
        return dict(found, backend="sqlite-fallback") if found else None

    def save_model_version(self, record: dict[str, Any]) -> dict[str, Any]:
        try:
            self._call("mutation", "ctaid:saveModelVersion", {"record": record})
            return dict(record, backend=self.name)
        except Exception:
            return dict(self.fallback.save_model_version(record), backend="sqlite-fallback")

    def list_model_versions(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            value = self._call("query", "ctaid:listModelVersions", {"limit": int(limit)})
            if isinstance(value, list):
                return [dict(item, backend=self.name) for item in value]
        except Exception:
            pass
        return [dict(item, backend="sqlite-fallback") for item in self.fallback.list_model_versions(limit)]

    def save_prediction(self, record: dict[str, Any]) -> dict[str, Any]:
        try:
            self._call("mutation", "ctaid:savePrediction", {"record": record})
            return dict(record, backend=self.name)
        except Exception:
            return dict(self.fallback.save_prediction(record), backend="sqlite-fallback")

    def list_predictions(self, limit: int = 50) -> list[dict[str, Any]]:
        try:
            value = self._call("query", "ctaid:listPredictions", {"limit": int(limit)})
            if isinstance(value, list):
                return [dict(item, backend=self.name) for item in value]
        except Exception:
            pass
        return [dict(item, backend="sqlite-fallback") for item in self.fallback.list_predictions(limit)]

    def count_predictions(self) -> int:
        return len(self.list_predictions(limit=500))

    def reset(self) -> None:
        # The deployment may be unreachable; the local mirror is always cleared.
        with contextlib.suppress(Exception):
            self._call("mutation", "ctaid:resetDemo", {})
        self.fallback.reset()


def build_backend(config: ConvexConfig, sqlite_path: Path):
    """Return the Convex backend when a URL is set, otherwise SQLite."""
    sqlite = SQLiteBackend(sqlite_path)
    if config.enabled:
        return ConvexBackend(config, sqlite)
    return sqlite


@dataclass(frozen=True)
class BackendStatus:
    backend: str
    dataset_saved: bool
    prediction_count: int
    model_versions: int
    latency_ms: int


def status(backend, dataset_meta: dict[str, Any] | None = None) -> BackendStatus:
    started = time.perf_counter()
    predictions = backend.list_predictions(limit=100)
    versions = backend.list_model_versions(limit=100)
    latency = int((time.perf_counter() - started) * 1000)
    return BackendStatus(
        backend=backend.name,
        dataset_saved=dataset_meta is not None,
        prediction_count=len(predictions),
        model_versions=len(versions),
        latency_ms=latency,
    )
