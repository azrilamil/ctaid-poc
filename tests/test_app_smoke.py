"""Smoke tests for the Streamlit app: render, route, train and score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _app() -> AppTest:
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=300)


def test_home_page_renders() -> None:
    app = _app()
    app.run()
    assert not app.exception
    assert not app.error
    body = " ".join(block.value for block in app.markdown if block.value)
    assert "CTAID" in body
    assert "Demo with synthetic data, not a medical tool." in body


def test_every_page_renders() -> None:
    for page in ["Home", "Data Exploration", "Model Training", "Prediction"]:
        app = _app()
        app.run()
        app.radio(key="page").set_value(page).run()
        assert not app.exception, page
        assert not app.error, page


def test_training_click_produces_metrics() -> None:
    app = _app()
    app.run()
    app.radio(key="page").set_value("Model Training").run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) >= 2, "expected a metrics table and a classification report"


def test_prediction_page_offers_input_methods() -> None:
    app = _app()
    app.run()
    app.radio(key="page").set_value("Prediction").run()
    assert not app.exception, app.exception
    options = [str(radio.options) for radio in app.radio]
    assert any("Generated sample" in value for value in options)


def test_prediction_scores_and_logs_a_generated_sample() -> None:
    app = _app()
    app.run()
    app.radio(key="page").set_value("Prediction").run()
    app.radio[0].set_value("Generated sample").run()
    assert not app.exception, app.exception
    log_buttons = [button for button in app.button if button.label.startswith("Log")]
    assert log_buttons, "expected a log button on the result panel"
    log_buttons[0].click().run()
    assert not app.exception, app.exception
    assert any("Prediction logged" in item.value for item in app.success)


def test_dataset_summary_values() -> None:
    from app.data import summary
    from scripts.generate_synthetic_data import build_frame

    info = summary(build_frame(records=50))
    assert info.rows == 150
    assert info.feature_count == 65
    assert sum(info.class_counts.values()) == 150
    assert np.isclose(sum(info.class_balance.values()), 1.0)
