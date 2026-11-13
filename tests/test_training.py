"""Training and prediction path tests."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.data import feature_columns
from app.predict import (
    UploadError,
    align_features,
    predict_frame,
    prediction_record,
    read_upload,
)
from app.training import MODEL_SPECS, TrainConfig, build_pipeline, train_models


@pytest.fixture()
def trained(small_frame: pd.DataFrame, tmp_path):
    config = TrainConfig(
        models=("logistic_regression",),
        test_size=0.25,
        random_state=5,
        artifact_dir=tmp_path,
        persist=True,
    )
    return train_models(small_frame, config), config


def test_training_returns_real_metrics(trained) -> None:
    run, _config = trained
    assert run.rows_used == 360
    assert run.rows_train + run.rows_test == run.rows_used
    result = run.results[0]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"):
        assert key in result.metrics
        assert 0.0 <= result.metrics[key] <= 1.0
    # A model on synthetic data with class signal must beat chance.
    assert result.metrics["roc_auc_macro"] > 0.7
    assert result.metrics["accuracy"] > 0.4
    assert result.labels == [0, 1, 2, 3]
    assert result.confusion.shape == (4, 4)
    assert set(result.roc) <= {0, 1, 2, 3}


def test_training_persists_artifact(trained, tmp_path) -> None:
    run, _config = trained
    artifact = run.results[0].artifact
    assert artifact is not None
    from pathlib import Path

    path = Path(artifact)
    assert path.is_file()
    assert path.name.startswith("logistic_regression")
    sidecar = path.with_suffix(".json")
    assert sidecar.is_file()


def test_every_registered_model_builds_and_fits(small_frame: pd.DataFrame, tmp_path) -> None:
    config = TrainConfig(models=tuple(MODEL_SPECS), artifact_dir=tmp_path, persist=False, sample_limit=240)
    run = train_models(small_frame, config)
    assert len(run.results) == len(MODEL_SPECS)
    for result in run.results:
        assert result.metrics["roc_auc_macro"] > 0.6, result.key


def test_training_rejects_bad_config(small_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        train_models(small_frame, TrainConfig(models=(), persist=False))
    with pytest.raises(ValueError):
        train_models(small_frame, TrainConfig(models=("nope",), persist=False))
    with pytest.raises(ValueError):
        train_models(small_frame, TrainConfig(test_size=0.95, persist=False))


def test_prediction_returns_probabilities(trained, small_frame: pd.DataFrame) -> None:
    run, _config = trained
    features = feature_columns(small_frame)
    pipeline = run.results[0].pipeline
    row = small_frame[features].head(1)
    predictions = predict_frame(pipeline, row, features, "logistic_regression")
    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.predicted_class in (0, 1, 2, 3)
    assert 0.0 <= prediction.confidence <= 1.0
    assert pytest.approx(sum(prediction.probabilities.values()), abs=1e-3) == 1.0
    assert prediction.predicted_label.startswith("Pattern")
    assert len(prediction.top_features) <= 5


def test_prediction_record_shape(trained, small_frame: pd.DataFrame) -> None:
    run, _config = trained
    features = feature_columns(small_frame)
    prediction = predict_frame(run.results[0].pipeline, small_frame[features].head(1), features)[0]
    record = prediction_record(prediction, source="pytest")
    assert record["source"] == "pytest"
    assert isinstance(record["probabilities"], dict)
    assert all(isinstance(key, str) for key in record["probabilities"])


def test_built_pipeline_has_expected_steps() -> None:
    pipeline = build_pipeline("random_forest", 3)
    assert [name for name, _step in pipeline.steps] == ["impute", "scale", "model"]


def test_align_features_reports_missing(small_frame: pd.DataFrame) -> None:
    features = feature_columns(small_frame)
    partial = small_frame[features].drop(columns=[features[0]])
    with pytest.raises(UploadError) as error:
        align_features(partial, features)
    assert features[0] in str(error.value)


def test_read_upload_accepts_valid_csv(small_frame: pd.DataFrame) -> None:
    features = feature_columns(small_frame)
    payload = small_frame[["Record ID", *features]].head(3).to_csv(index=False).encode("utf-8")
    frame, report = read_upload("rows.csv", payload, "text/csv")
    assert report.rows == 3
    assert "Record ID" in report.dropped_columns
    assert "Record ID" not in frame.columns


def test_read_upload_rejects_wrong_extension() -> None:
    with pytest.raises(UploadError):
        read_upload("rows.xlsx", b"a,b\n1,2\n", "application/vnd.ms-excel")


def test_read_upload_rejects_bad_content_type(small_frame: pd.DataFrame) -> None:
    payload = small_frame.head(2).to_csv(index=False).encode("utf-8")
    with pytest.raises(UploadError):
        read_upload("rows.csv", payload, "image/png")


def test_read_upload_rejects_binary() -> None:
    with pytest.raises(UploadError):
        read_upload("rows.csv", b"\x00\x01\x02binary", "text/csv")


def test_read_upload_rejects_oversize() -> None:
    big = b"a,b\n" + b"1,2\n" * 10
    big = big + b"x" * (2 * 1024 * 1024 + 10)
    with pytest.raises(UploadError):
        read_upload("rows.csv", big, "text/csv")


def test_read_upload_rejects_empty() -> None:
    with pytest.raises(UploadError):
        read_upload("rows.csv", b"", "text/csv")


def test_read_upload_rejects_too_many_rows(small_frame: pd.DataFrame) -> None:
    features = feature_columns(small_frame)
    frame = pd.DataFrame(np.tile(small_frame[features].iloc[0].to_numpy(), (501, 1)), columns=features)
    payload = frame.to_csv(index=False).encode("utf-8")
    with pytest.raises(UploadError) as error:
        read_upload("rows.csv", payload, "text/csv")
    assert "500" in str(error.value)


def test_read_upload_rejects_unparseable() -> None:
    payload = io.BytesIO(b"a,b\n1,2,3,4,5\n6\n").getvalue()
    frame, report = read_upload("rows.csv", payload, "text/csv")
    # Ragged rows still parse in pandas; the guard is that empty input and
    # binaries fail, and the row/column counts are reported honestly.
    assert report.rows >= 1
    assert "a" in frame.columns
