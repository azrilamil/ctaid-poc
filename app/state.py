"""Cached accessors shared by the pages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.backend import build_backend
from app.config import Settings
from app.data import load_dataset, summary
from app.storage import build_store


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings.load()


@st.cache_resource(show_spinner=False)
def get_dataset_meta() -> dict:
    """Register the synthetic dataset with the backend once per process."""
    from app.bootstrap import ensure_dataset_meta

    try:
        return ensure_dataset_meta(get_settings())
    except Exception as exc:  # pragma: no cover - bootstrap guard
        return {"name": "synthetic_features.csv", "rows": 0, "error": str(exc)}


@st.cache_data(show_spinner=False)
def get_dataset(path: str, mtime: float) -> pd.DataFrame:
    """Load the synthetic table, keyed on path and modification time."""
    return load_dataset(Path(path))


def dataset() -> pd.DataFrame:
    settings = get_settings()
    path = settings.dataset_path
    mtime = path.stat().st_mtime if path.is_file() else 0.0
    return get_dataset(str(path), mtime)


@st.cache_data(show_spinner=False)
def dataset_summary(path: str, mtime: float) -> dict:
    frame = get_dataset(path, mtime)
    return summary(frame).__dict__


def current_summary() -> dict:
    settings = get_settings()
    path = settings.dataset_path
    mtime = path.stat().st_mtime if path.is_file() else 0.0
    return dataset_summary(str(path), mtime)


@st.cache_resource(show_spinner=False)
def get_backend():
    settings = get_settings()
    return build_backend(settings.convex, settings.state_dir / "ctaid-demo.sqlite3")


def backend():
    return get_backend()


@st.cache_resource(show_spinner=False)
def get_store():
    settings = get_settings()
    return build_store(settings.storage, settings.state_dir / "objects")


def store():
    return get_store()


def model_dir() -> Path:
    path = get_settings().models_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def trained_runs() -> dict:
    """Hold fitted pipelines for the session (never cached across users)."""
    if "runs" not in st.session_state:
        st.session_state["runs"] = {}
    return st.session_state["runs"]


def active_run():
    runs = trained_runs()
    key = st.session_state.get("active_run")
    return runs.get(key) if key else None


def set_active_run(run_id: str) -> None:
    st.session_state["active_run"] = run_id
