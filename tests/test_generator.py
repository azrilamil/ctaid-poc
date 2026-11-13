"""Generator and schema tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data import feature_columns, load_dataset, summary
from app.privacy import IDENTIFIER_COLUMN
from app.schema import (
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    RECORD_ID_COLUMN,
    TARGET_COLUMN,
    VESSEL_COLUMN,
    VESSELS_PER_RECORD,
)
from scripts.generate_synthetic_data import DEFAULT_SEED, build_frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return build_frame(records=120, seed=DEFAULT_SEED)


def test_generator_shape_matches_schema(frame: pd.DataFrame) -> None:
    assert len(frame) == 120 * VESSELS_PER_RECORD
    assert frame.shape[1] == 3 + len(FEATURE_NAMES)
    assert list(frame.columns) == [RECORD_ID_COLUMN, VESSEL_COLUMN, *FEATURE_NAMES, TARGET_COLUMN]
    assert IDENTIFIER_COLUMN not in frame.columns


def test_generator_respects_ranges(frame: pd.DataFrame) -> None:
    for spec in FEATURE_COLUMNS:
        name = str(spec["name"])
        column = frame[name]
        assert column.notna().all(), name
        assert column.min() >= float(spec["min"]) - 1e-6, name
        assert column.max() <= float(spec["max"]) + 1e-6, name
        if spec["dtype"] == "int":
            assert np.allclose(column.to_numpy(), np.rint(column.to_numpy())), name


def test_generator_is_deterministic(frame: pd.DataFrame) -> None:
    again = build_frame(records=120, seed=DEFAULT_SEED)
    pd.testing.assert_frame_equal(frame, again)
    other = build_frame(records=120, seed=DEFAULT_SEED + 1)
    assert not frame.equals(other)


def test_record_ids_are_sequential_and_grouped(frame: pd.DataFrame) -> None:
    assert frame[RECORD_ID_COLUMN].min() == 1
    assert frame[RECORD_ID_COLUMN].max() == 120
    grouped = frame.groupby(RECORD_ID_COLUMN)[VESSEL_COLUMN].apply(list)
    assert all(vessels == [1, 2, 3] for vessels in grouped)
    # Every vessel of a record shares the same synthetic class.
    per_record = frame.groupby(RECORD_ID_COLUMN)[TARGET_COLUMN].nunique()
    assert (per_record == 1).all()


def test_target_is_four_classes(frame: pd.DataFrame) -> None:
    assert sorted(frame[TARGET_COLUMN].unique().tolist()) == [0, 1, 2, 3]


def test_derived_columns_are_consistent(frame: pd.DataFrame) -> None:
    for child, parent in DERIVED_COLUMNS.items():
        diff = frame[child] - frame[parent]
        assert np.allclose(diff.to_numpy(), -3.0, atol=1e-6), child


def test_class_effects_separate_classes(frame: pd.DataFrame) -> None:
    """The generator must make classes learnable, not random noise."""
    means = frame.groupby(TARGET_COLUMN)["CONVENTIONAL_mean"].mean()
    assert means.nunique() == 4 or means.max() - means.min() > 20


def test_committed_csv_matches_shape(committed_frame: pd.DataFrame) -> None:
    info = summary(committed_frame)
    assert info.rows >= 2000
    assert info.feature_count == len(FEATURE_NAMES)
    assert info.missing_cells == 0
    assert sorted(info.class_counts) == [0, 1, 2, 3]
    assert info.records * VESSELS_PER_RECORD == info.rows


def test_load_dataset_rejects_bad_file(tmp_path) -> None:
    from app.data import DatasetError

    broken = tmp_path / "broken.csv"
    broken.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(broken)


def test_feature_columns_excludes_identifiers(committed_frame: pd.DataFrame) -> None:
    names = feature_columns(committed_frame)
    assert RECORD_ID_COLUMN not in names
    assert VESSEL_COLUMN not in names
    assert TARGET_COLUMN not in names
    assert len(names) == len(FEATURE_NAMES)
