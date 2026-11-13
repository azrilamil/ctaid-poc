"""Backend, storage and artwork tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.artwork import build_svg, ensure_hero
from app.backend import SQLiteBackend, build_backend, status
from app.config import ConvexConfig, ObjectStoreConfig
from app.storage import LocalObjectStore, build_store


def test_sqlite_backend_round_trips(tmp_path: Path) -> None:
    backend = SQLiteBackend(tmp_path / "demo.sqlite3")
    backend.save_dataset_meta(
        {
            "name": "synthetic_features.csv",
            "rows": 2100,
            "columns": 68,
            "records": 700,
            "class_counts": {"0": 597, "1": 552, "2": 513, "3": 438},
            "digest": "abc123",
        }
    )
    meta = backend.latest_dataset_meta()
    assert meta is not None
    assert meta["rows"] == 2100

    backend.save_prediction(
        {
            "source": "pytest",
            "predicted_class": 2,
            "predicted_label": "Pattern C",
            "confidence": 0.71,
            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.71, "3": 0.09},
            "top_features": ["a", "b"],
        }
    )
    backend.save_model_version(
        {
            "model": "logistic_regression",
            "label": "Logistic Regression",
            "metrics": {"accuracy": 0.7},
            "rows_used": 900,
            "feature_count": 65,
            "artifact": "models/x.joblib",
        }
    )

    predictions = backend.list_predictions()
    assert len(predictions) == 1
    assert predictions[0]["probabilities"]["2"] == pytest.approx(0.71)
    versions = backend.list_model_versions()
    assert versions[0]["metrics"]["accuracy"] == pytest.approx(0.7)

    snapshot = status(backend, meta)
    assert snapshot.backend == "sqlite"
    assert snapshot.prediction_count == 1
    assert snapshot.model_versions == 1

    backend.reset()
    assert backend.list_predictions() == []
    assert backend.list_model_versions() == []
    assert backend.latest_dataset_meta() is not None


def test_sqlite_schema_is_created(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "demo.sqlite3"
    SQLiteBackend(path)
    conn = sqlite3.connect(path)
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert {"dataset_meta", "model_versions", "predictions"} <= tables


def test_build_backend_defaults_to_sqlite(tmp_path: Path) -> None:
    backend = build_backend(ConvexConfig(), tmp_path / "demo.sqlite3")
    assert backend.name == "sqlite"


def test_convex_backend_falls_back_when_unreachable(tmp_path: Path) -> None:
    config = ConvexConfig(url="http://127.0.0.1:9", timeout_s=0.5)
    backend = build_backend(config, tmp_path / "demo.sqlite3")
    assert backend.name == "convex"
    saved = backend.save_prediction(
        {
            "source": "pytest",
            "predicted_class": 1,
            "predicted_label": "Pattern B",
            "confidence": 0.6,
            "probabilities": {"1": 0.6},
            "top_features": [],
        }
    )
    assert saved["backend"] == "sqlite-fallback"
    assert backend.count_predictions() == 1


def test_local_store_put_get_list(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    stored = store.put_bytes("uploads/a.csv", b"Record ID\n1\n", "text/csv")
    assert stored.backend == "local"
    assert store.exists("uploads/a.csv")
    assert store.get_bytes("uploads/a.csv").startswith(b"Record ID")
    assert store.list_keys("uploads/") == ["uploads/a.csv"]
    store.delete("uploads/a.csv")
    assert not store.exists("uploads/a.csv")
    assert store.list_keys() == []


def test_local_store_blocks_traversal(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    with pytest.raises(ValueError):
        store.put_bytes("../../escape.csv", b"x")


def test_local_store_clear(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    store.put_bytes("a/one.txt", b"1")
    store.put_bytes("b/two.txt", b"2")
    assert store.clear() == 2
    assert store.list_keys() == []


def test_build_store_defaults_to_local(tmp_path: Path) -> None:
    store = build_store(ObjectStoreConfig(), tmp_path / "objects")
    assert store.backend == "local"


def test_s3_store_falls_back_when_endpoint_is_dead(tmp_path: Path) -> None:
    config = ObjectStoreConfig(
        endpoint="http://127.0.0.1:9",
        bucket="ctaid-poc-demo",
        access_key="demo",
        secret_key="demo",
    )
    store = build_store(config, tmp_path / "objects")
    stored = store.put_bytes("uploads/x.csv", b"a,b\n1,2\n", "text/csv")
    assert stored.backend == "local-fallback"
    assert store.get_bytes("uploads/x.csv") == b"a,b\n1,2\n"


def test_hero_svg_is_original_and_valid() -> None:
    svg = build_svg()
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "Generated illustration, not a scan of a person" in svg
    assert "<image" not in svg
    assert "http" not in svg.replace("http://www.w3.org/2000/svg", "")
    assert "aria-label" in svg


def test_ensure_hero_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import artwork

    monkeypatch.setattr(artwork, "ASSETS_DIR", tmp_path)
    path = ensure_hero()
    assert path.is_file()
    assert path.read_text(encoding="utf-8").startswith("<svg")


def test_convex_http_payload_keeps_format_out_of_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Convex HTTP API rejects unknown fields inside args with a server error.

    `format` belongs on the envelope, not inside the function arguments.
    """
    import sys
    import types

    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "success", "value": [{"model": "random_forest"}]}

    def _post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Resp()

    fake = types.ModuleType("requests")
    fake.post = _post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)

    from app.backend import ConvexBackend, SQLiteBackend
    from app.config import ConvexConfig

    backend = ConvexBackend(
        ConvexConfig(url="https://convex-demo.azril.io", timeout_s=3),
        SQLiteBackend(tmp_path / "demo.sqlite3"),
    )
    versions = backend.list_model_versions(limit=5)
    assert versions[0]["model"] == "random_forest"
    assert versions[0]["backend"] == "convex"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["path"] == "ctaid:listModelVersions"
    assert payload["args"] == {"limit": 5}
    assert "format" not in payload["args"]
    assert payload["format"] == "json"
    assert captured["url"] == "https://convex-demo.azril.io/api/query"
