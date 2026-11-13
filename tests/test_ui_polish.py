"""Visitor facing polish: Pattern labels, default model, storage badges."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data import feature_columns
from app.labels import (
    DEMO_STORAGE_HINT,
    DEMO_STORAGE_LABEL,
    artifact_badge,
    metadata_badge,
)
from app.pages_data import _display_frame
from app.pages_prediction import DEFAULT_MODEL_KEY, _artifact_label, default_model_label
from app.predict import feature_weights, predict_frame
from app.schema import DISPLAY_LABELS, TARGET_COLUMN, TARGET_DISPLAY
from app.training import TrainConfig, train_models


# 1. Pattern instead of the internal column name -------------------------
def test_target_display_alias_is_pattern() -> None:
    assert TARGET_DISPLAY == "Pattern"
    assert DISPLAY_LABELS[TARGET_COLUMN] == "Pattern"


def test_display_frame_renames_only_the_target_column(small_frame: pd.DataFrame) -> None:
    features = feature_columns(small_frame)
    shown = _display_frame(small_frame[["Vessel", *features[:3], TARGET_COLUMN]])
    assert "Pattern" in shown.columns
    assert TARGET_COLUMN not in shown.columns
    assert "Vessel" in shown.columns
    assert list(shown[features[:3]].columns) == features[:3]


# 2. Random Forest default and real importances ---------------------------
def test_default_picker_prefers_random_forest() -> None:
    models = {
        "Logistic Regression (saved artifact)": {"key": "logistic_regression", "metrics": {"roc_auc_macro": 0.91}},
        "Random Forest (saved artifact)": {"key": "random_forest", "metrics": {"roc_auc_macro": 0.95}},
        "Support Vector Machine (saved artifact)": {"key": "svm", "metrics": {"roc_auc_macro": 0.90}},
    }
    assert default_model_label(models) == "Random Forest (saved artifact)"
    assert DEFAULT_MODEL_KEY == "random_forest"


def test_default_picker_uses_best_random_forest_when_several_exist() -> None:
    models = {
        "Random Forest, AUC 0.940 (saved artifact)": {"key": "random_forest", "metrics": {"roc_auc_macro": 0.94}},
        "Random Forest, AUC 0.960 (saved artifact)": {"key": "random_forest", "metrics": {"roc_auc_macro": 0.96}},
        "Logistic Regression (saved artifact)": {"key": "logistic_regression", "metrics": {"roc_auc_macro": 0.92}},
    }
    assert default_model_label(models) == "Random Forest, AUC 0.960 (saved artifact)"


def test_default_picker_falls_back_to_first_option() -> None:
    models = {"Logistic Regression (saved artifact)": {"key": "logistic_regression", "metrics": {}}}
    assert default_model_label(models) == "Logistic Regression (saved artifact)"
    assert default_model_label({}) is None


def test_artifact_label_quotes_saved_metrics() -> None:
    label = _artifact_label("Random Forest", {"roc_auc_macro": 0.9595, "accuracy": 0.8419})
    assert label == "Random Forest, AUC 0.960, accuracy 0.842 (saved artifact)"
    assert _artifact_label("Random Forest", {}) == "Random Forest (saved artifact)"


@pytest.fixture()
def forest_run(small_frame: pd.DataFrame, tmp_path):
    config = TrainConfig(models=("random_forest",), random_state=3, artifact_dir=tmp_path, persist=True)
    return train_models(small_frame, config)


def test_random_forest_importance_is_not_all_zero(forest_run, small_frame: pd.DataFrame) -> None:
    features = feature_columns(small_frame)
    pipeline = forest_run.results[0].pipeline
    weights = feature_weights(pipeline, features)
    assert len(weights) == len(features)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert max(weights.values()) > 0.0


def test_linear_model_importance_is_absolute_coefficients(forest_run, small_frame: pd.DataFrame) -> None:
    features = feature_columns(small_frame)
    config = TrainConfig(models=("logistic_regression",), random_state=3, persist=False)
    run = train_models(small_frame, config)
    weights = feature_weights(run.results[0].pipeline, features)
    assert weights
    assert all(value >= 0.0 for value in weights.values())
    assert max(weights.values()) > 0.0


def test_prediction_top_features_are_non_zero(forest_run, small_frame: pd.DataFrame) -> None:
    features = feature_columns(small_frame)
    pipeline = forest_run.results[0].pipeline
    prediction = predict_frame(pipeline, small_frame[features].head(1), features, "random_forest")[0]
    assert prediction.top_features
    assert all(weight > 0.0 for _name, _value, weight in prediction.top_features)
    assert 0.0 <= prediction.confidence <= 1.0


# 3. Storage badges -------------------------------------------------------
def test_local_backends_show_demo_storage() -> None:
    for name in ("sqlite", "local", "local-fallback", "unknown"):
        assert metadata_badge(name) == (DEMO_STORAGE_LABEL, DEMO_STORAGE_HINT)
        assert artifact_badge(name) == (DEMO_STORAGE_LABEL, DEMO_STORAGE_HINT)


def test_configured_backends_name_the_product() -> None:
    assert metadata_badge("convex")[0] == "Convex"
    assert artifact_badge("garage-s3")[0] == "Garage S3"
    assert DEMO_STORAGE_LABEL == "Demo storage"


def test_upload_drops_pattern_alias_and_reports_friendly_name(small_frame: pd.DataFrame) -> None:
    from app.predict import read_upload

    features = feature_columns(small_frame)
    frame = small_frame[features[:4]].head(2).copy()
    frame["Pattern"] = 1
    payload = frame.to_csv(index=False).encode("utf-8")
    parsed, report = read_upload("rows.csv", payload, "text/csv")
    assert "Pattern" in report.dropped_columns
    assert "Pattern" not in parsed.columns
    assert "Dropped non modelling column(s): Pattern." in report.warnings


def test_no_diagnosis_word_in_ui_copy() -> None:
    """Visitor facing copy must never carry the internal column name as a label."""
    from pathlib import Path

    from app import pages_data, pages_home, pages_prediction, pages_training, theme

    for module in (pages_data, pages_home, pages_prediction, pages_training, theme):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "diagnosis" not in source.lower(), module.__name__
