"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_DATASET  # noqa: E402
from app.data import load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def committed_frame() -> pd.DataFrame:
    return load_dataset(DEFAULT_DATASET)


@pytest.fixture()
def small_frame(committed_frame: pd.DataFrame) -> pd.DataFrame:
    return committed_frame.sample(n=360, random_state=11).reset_index(drop=True)
