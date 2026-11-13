"""Model Training page. Training really runs, nothing here is faked."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app import charts
from app.backend import status
from app.data import class_label
from app.labels import artifact_badge, metadata_badge, storage_caption
from app.schema import CLASS_LABELS
from app.state import backend, dataset, model_dir, set_active_run, store, trained_runs
from app.theme import MUTED, footer, page_header, section, stat
from app.training import DEFAULT_MODELS, MODEL_SPECS, TrainConfig, TrainingRun, train_models

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "roc_auc_macro": "ROC AUC (macro)",
}


def _run_id(run: TrainingRun) -> str:
    return f"{run.finished_at}-{'-'.join(result.key for result in run.results)}"


def _persist(run: TrainingRun, active_backend, active_store) -> list[str]:
    """Push run metadata to Convex/SQLite and artifacts to Garage/local disk."""
    notes: list[str] = []
    for result in run.results:
        active_backend.save_model_version(
            {
                "model": result.key,
                "label": result.label,
                "metrics": result.metrics,
                "rows_used": run.rows_used,
                "feature_count": len(run.feature_names),
                "artifact": result.artifact,
            }
        )
        if result.artifact:
            try:
                stored = active_store.put_file(
                    f"models/{result.key}.joblib",
                    __import__("pathlib").Path(result.artifact),
                    content_type="application/octet-stream",
                )
                label, _hint = artifact_badge(stored.backend)
                notes.append(f"{result.label}: artifact in {label}")
            except Exception as exc:  # pragma: no cover - storage guard
                notes.append(f"{result.label}: artifact kept locally ({exc})")
    return notes


def _controls():
    left, right = st.columns([1.3, 1])
    with left:
        chosen = st.multiselect(
            "Models to train",
            options=list(MODEL_SPECS),
            default=list(DEFAULT_MODELS),
            format_func=lambda key: MODEL_SPECS[key]["label"],
            help="Each model gets the same split so the comparison is fair.",
        )
    with right:
        test_size = st.slider("Test split", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
    with st.expander("Advanced options"):
        left, right = st.columns(2)
        with left:
            random_state = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
        with right:
            row_cap = st.number_input(
                "Max rows used",
                min_value=300,
                max_value=3000,
                value=1400,
                step=100,
                help="Keep training fast on a demo host. The full table is 2100 rows.",
            )
    return tuple(chosen), float(test_size), int(random_state), int(row_cap)


def _run_training(models, test_size, random_state, row_cap) -> None:
    frame = dataset()
    config = TrainConfig(
        models=tuple(models),
        test_size=test_size,
        random_state=random_state,
        sample_limit=row_cap,
        artifact_dir=model_dir(),
        persist=True,
    )
    with st.status("Training on synthetic data", expanded=True) as progress:
        progress.write(f"Rows in table: {len(frame):,}. Cap: {row_cap:,}.")
        progress.write("Fitting " + ", ".join(MODEL_SPECS[key]["label"] for key in models))
        run = train_models(frame, config)
        run_id = _run_id(run)
        trained_runs()[run_id] = run
        set_active_run(run_id)
        progress.write("Writing joblib artifacts to models/")
        notes = _persist(run, backend(), store())
        for note in notes:
            progress.write(note)
        progress.update(label="Training complete", state="complete", expanded=False)


def _metrics_table(run: TrainingRun) -> pd.DataFrame:
    rows = []
    for result in run.results:
        rows.append(
            {
                "model": result.label,
                **{METRIC_LABELS[key]: value for key, value in result.metrics.items()},
                "train seconds": round(result.train_seconds, 3),
            }
        )
    table = pd.DataFrame(rows)
    return table.round(4)


def _report_table(result) -> pd.DataFrame:
    report = result.report
    rows = []
    for label in result.labels:
        block = report.get(str(label), {})
        rows.append(
            {
                "class": class_label(label),
                "precision": round(block.get("precision", 0.0), 4),
                "recall": round(block.get("recall", 0.0), 4),
                "f1-score": round(block.get("f1-score", 0.0), 4),
                "support": int(block.get("support", 0)),
            }
        )
    macro = report.get("macro avg", {})
    rows.append(
        {
            "class": "macro avg",
            "precision": round(macro.get("precision", 0.0), 4),
            "recall": round(macro.get("recall", 0.0), 4),
            "f1-score": round(macro.get("f1-score", 0.0), 4),
            "support": int(macro.get("support", 0)),
        }
    )
    return pd.DataFrame(rows)


def _render_results(run: TrainingRun) -> None:
    cols = st.columns(4)
    with cols[0]:
        stat("Rows used", f"{run.rows_used:,}", f"train {run.rows_train:,} / test {run.rows_test:,}")
    with cols[1]:
        stat("Features", f"{len(run.feature_names)}", "shared across models")
    with cols[2]:
        stat("Classes", f"{len(run.labels)}", "all present in the split")
    with cols[3]:
        best = run.best()
        stat("Best ROC AUC", f"{best.metrics['roc_auc_macro']:.3f}", best.label)

    section("Metric comparison", "Held out split, macro averages over the four classes.")
    st.dataframe(_metrics_table(run), width="stretch", hide_index=True)
    per_model = {result.label: result.metrics for result in run.results}
    st.plotly_chart(charts.metric_bars(per_model, list(METRIC_LABELS)), width="stretch", theme=None)

    section("Per model detail", "Classification report, confusion matrix and ROC curves.")
    labels = [result.label for result in run.results]
    tabs = st.tabs(labels)
    for tab, result in zip(tabs, run.results, strict=True):
        with tab:
            head = st.columns(5)
            for column, (key, value) in zip(head, result.metrics.items(), strict=True):
                with column:
                    stat(METRIC_LABELS.get(key, key), f"{value:.3f}")
            left, right = st.columns([1, 1])
            with left:
                st.dataframe(_report_table(result), width="stretch", hide_index=True)
                st.caption(MODEL_SPECS[result.key]["note"])
            with right:
                st.plotly_chart(
                    charts.confusion(result.confusion, result.labels, f"{result.label} confusion matrix"),
                    width="stretch",
                    theme=None,
                )
            st.plotly_chart(
                charts.roc(result.roc, f"{result.label} ROC curves (one versus rest)"),
                width="stretch",
                theme=None,
            )
            st.plotly_chart(
                charts.importance(result.importance, f"{result.label} feature importance"),
                width="stretch",
                theme=None,
            )
            if result.artifact:
                st.caption(f"Saved as {Path(result.artifact).name} in the demo model store.")

    section("Version log", "Model versions recorded by the backend for this demo session.")
    versions = backend().list_model_versions(limit=12)
    if versions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": item.get("label", item.get("model")),
                        "accuracy": round(float(item.get("metrics", {}).get("accuracy", 0.0)), 4),
                        "roc auc": round(float(item.get("metrics", {}).get("roc_auc_macro", 0.0)), 4),
                        "rows": item.get("rows_used"),
                        "logged at": item.get("created_at"),
                    }
                    for item in versions
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No model versions recorded yet.")


def render() -> None:
    page_header(
        "Model Training",
        "Fit and compare classifiers",
        "Training runs for real with scikit-learn on the synthetic table. Metrics are "
        "measured on a held out split, the fitted pipeline is saved to models/ as a joblib "
        "artifact, and the run is logged to the backend.",
    )

    info = status(backend())
    metadata_label, metadata_hint = metadata_badge(info.backend)
    storage = store()
    storage_label, storage_hint = artifact_badge(storage.backend)
    cols = st.columns(4)
    with cols[0]:
        stat("Metadata backend", metadata_label, metadata_hint)
    with cols[1]:
        stat("Model versions", f"{info.model_versions}", "recorded runs")
    with cols[2]:
        stat("Predictions logged", f"{info.prediction_count}", "synthetic only")
    with cols[3]:
        stat("Artifact store", storage_label, storage_hint)
    caption = storage_caption(metadata_label, storage_label)
    if caption:
        st.caption(caption)

    section("Configuration", "Pick models, split size and an optional row cap.")
    models, test_size, random_state, row_cap = _controls()

    start = st.button("Train selected models", type="primary", disabled=not models)
    if not models:
        st.warning("Select at least one model.")

    if start:
        try:
            _run_training(models, test_size, random_state, row_cap)
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    run = None
    runs = trained_runs()
    if runs:
        keys = list(runs)
        chosen = st.selectbox("Session run", options=keys, index=len(keys) - 1)
        run = runs[chosen]

    if run is None:
        st.markdown(
            f'<p style="color:{MUTED};margin-top:1rem;">No run in this session yet. '
            "Choose models above and press Train to produce real metrics.</p>",
            unsafe_allow_html=True,
        )
        for class_id in CLASS_LABELS:
            st.caption(f"{class_label(class_id)} is an invented pattern group, not a clinical finding.")
        footer()
        return

    _render_results(run)
    footer()
