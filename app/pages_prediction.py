"""Prediction page: form, generated sample class, or small CSV upload."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts
from app.data import class_caption, class_label, feature_columns, sample_for_class
from app.predict import UploadError, align_features, predict_frame, prediction_record, read_upload
from app.schema import TARGET_COLUMN
from app.state import backend, dataset, store, trained_runs
from app.theme import MUTED, footer, notice, page_header, section, stat
from app.training import MODEL_SPECS, load_artifact

KEY_FEATURES = [
    "CONVENTIONAL_mean",
    "CONVENTIONAL_std",
    "CONVENTIONAL_max",
    "CONVENTIONAL_calciumAgatstonScore[onlyForCT]",
    "SHAPE_Volume(vx)",
    "GLCM_Contrast[=Variance]",
    "GLCM_Homogeneity[=InverseDifference]",
]


def _available_models() -> dict[str, dict]:
    """Fitted pipelines from this session, falling back to saved artifacts."""
    options: dict[str, dict] = {}
    for _run_id, run in trained_runs().items():
        for result in run.results:
            options[f"{result.label} (this session)"] = {
                "pipeline": result.pipeline,
                "feature_names": run.feature_names,
                "key": result.key,
            }
    if options:
        return options
    from app.state import model_dir

    names = feature_columns(dataset())
    for key, spec in MODEL_SPECS.items():
        try:
            pipeline = load_artifact(model_dir(), key)
        except Exception:
            continue
        options[f"{spec['label']} (saved artifact)"] = {
            "pipeline": pipeline,
            "feature_names": names,
            "key": key,
        }
    return options


def _model_picker(models: dict[str, dict]) -> dict | None:
    if not models:
        st.warning(
            "No trained model available. Open Model Training and press Train, "
            "or run python scripts/train_demo_model.py to create the default artifact."
        )
        return None
    choice = st.selectbox("Model", options=list(models))
    return models[choice]


def _inputs_from_form(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    medians = frame[features].median()
    st.caption(
        "Move the key features below. Every other column stays at the synthetic median so "
        "the demo stays quick to use."
    )
    values: dict[str, float] = {}
    for feature in KEY_FEATURES:
        if feature not in features:
            continue
        low = float(frame[feature].quantile(0.01))
        high = float(frame[feature].quantile(0.99))
        default = float(medians[feature])
        if low == high:
            values[feature] = st.number_input(feature, value=default)
            continue
        step = (high - low) / 200
        values[feature] = st.slider(
            feature,
            min_value=float(low),
            max_value=float(high),
            value=default,
            step=float(step),
            format="%.3f",
        )

    row = medians.to_dict()
    row.update(values)
    return pd.DataFrame([row])[list(features)]


def _inputs_from_class(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    classes = sorted(int(c) for c in frame[TARGET_COLUMN].unique())
    class_id = st.selectbox(
        "Generated sample pattern",
        options=classes,
        format_func=lambda value: f"{class_label(value)}: {class_caption(value)}",
    )
    seed = int(st.number_input("Sample draw seed", min_value=0, max_value=9999, value=7, step=1))
    row = sample_for_class(frame, class_id, seed=seed)
    st.caption(
        "A row drawn from the synthetic table, never from an original sample file. "
        "The class label is the generator's own label, shown for reference only."
    )
    with st.expander("Show the drawn feature values", expanded=False):
        st.dataframe(row[features].T.rename(columns={row.index[0]: "value"}), height=300)
    return row[features]


def _inputs_from_upload(features: list[str]) -> pd.DataFrame | None:
    uploaded = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
    st.caption("CSV only, 2 MB and 500 rows maximum. Identifier and label columns are dropped.")
    if uploaded is None:
        return None
    try:
        payload = uploaded.getvalue()
        frame, report = read_upload(uploaded.name, payload, uploaded.type)
    except UploadError as exc:
        st.error(str(exc))
        return None
    for warning in report.warnings:
        st.info(warning)
    if len(frame) > 1:
        st.warning(f"Upload has {len(frame)} rows. Only the first row is scored in this demo.")
    try:
        aligned, extra = align_features(frame.head(1), features)
    except UploadError as exc:
        st.error(str(exc))
        return None
    if extra:
        st.caption("Ignored extra column(s): " + ", ".join(extra[:6]))
    store().put_bytes(
        f"uploads/{report.name}",
        payload,
        content_type="text/csv",
    )
    st.success(f"Accepted {report.name}: {report.rows} rows, {report.columns} columns.")
    return aligned


def _render_result(prediction, frame) -> None:
    cols = st.columns(4)
    with cols[0]:
        stat("Predicted pattern", prediction.predicted_label, "synthetic class label")
    with cols[1]:
        stat("Confidence", f"{prediction.confidence * 100:.1f}%", "highest class probability")
    with cols[2]:
        stat("Model", MODEL_SPECS[prediction.model_key]["label"], "fitted on synthetic data")
    with cols[3]:
        stat("Rows scored", f"{prediction.rows_scored}", "this request")

    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(charts.probability_bars(prediction.probabilities), width="stretch", theme=None)
    with right:
        st.markdown(
            f"The model assigns **{prediction.predicted_label}** the highest probability. "
            "This is a number produced by a classifier trained only on invented data. "
            "It is not a diagnosis and must never inform care."
        )
        if prediction.top_features:
            st.markdown("**Features weighted most by this model:**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"feature": name, "value": round(value, 4), "importance": round(weight, 5)}
                        for name, value, weight in prediction.top_features
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


def render() -> None:
    frame = dataset()
    features = feature_columns(frame)

    page_header(
        "Prediction",
        "Score one synthetic case",
        "Enter feature values, load a generated pattern sample, or upload a small CSV. "
        "The app returns the most likely pattern class with class probabilities.",
    )

    models = _available_models()
    chosen = _model_picker(models)
    source = st.radio(
        "Input method",
        options=["Form", "Generated sample", "Upload CSV"],
        horizontal=True,
    )

    inputs = None
    if source == "Form":
        inputs = _inputs_from_form(frame, features)
    elif source == "Generated sample":
        inputs = _inputs_from_class(frame, features)
    else:
        inputs = _inputs_from_upload(features)

    if inputs is None or chosen is None:
        footer()
        return

    section("Result", "Probabilities come from the fitted pipeline, not a lookup table.")
    try:
        predictions = predict_frame(chosen["pipeline"], inputs, chosen["feature_names"], chosen["key"])
    except Exception as exc:
        st.error(f"Could not score this input: {exc}")
        footer()
        return

    prediction = predictions[0]
    _render_result(prediction, frame)

    if st.button("Log this prediction", type="secondary"):
        record = prediction_record(prediction, source=source.lower().replace(" ", "-"))
        backend().save_prediction(record)
        st.success("Prediction logged to the demo history (synthetic data only).")

    notice()
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.86rem;">Every logged prediction is synthetic demo '
        "traffic. There are no user accounts, and the daily reseed script clears this history.</p>",
        unsafe_allow_html=True,
    )
    footer()
