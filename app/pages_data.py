"""Data Exploration page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import charts
from app.data import (
    class_label,
    correlation_matrix,
    describe,
    feature_columns,
    top_correlations,
)
from app.schema import DISPLAY_LABELS, TARGET_COLUMN
from app.state import current_summary, dataset
from app.theme import MUTED, footer, page_header, section, stat, truncate


def _display_name(column: str) -> str:
    return DISPLAY_LABELS.get(column, column)


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename internal columns to their visitor facing labels."""
    return frame.rename(columns=DISPLAY_LABELS)


def _filter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(frame[TARGET_COLUMN].unique())
    chosen = st.multiselect(
        "Pattern classes",
        options=[int(c) for c in classes],
        default=[int(c) for c in classes],
        format_func=class_label,
        help="Narrow every chart on this page to the selected classes.",
    )
    if not chosen:
        st.info("Select at least one class to see data.")
        return frame.iloc[0:0]
    return frame[frame[TARGET_COLUMN].isin(chosen)]


def _table_view(frame: pd.DataFrame) -> None:
    features = feature_columns(frame)
    shown = [_display_name(column) for column in [TARGET_COLUMN, "Vessel", *features]]
    lookup = {_display_name(column): column for column in [TARGET_COLUMN, "Vessel", *features]}
    choice = st.selectbox(
        "Sort column",
        options=shown,
        index=0,
        help="Sort the synthetic table by any column.",
    )
    column = lookup[choice]
    ascending = st.toggle("Ascending", value=True)
    limit = st.slider("Rows to show", min_value=20, max_value=400, value=80, step=20)
    view = frame.sort_values(column, ascending=ascending).head(limit)
    st.dataframe(
        _display_frame(view),
        width="stretch",
        height=420,
        hide_index=True,
    )
    st.caption(
        f"{len(view)} of {len(frame):,} rows. Download below for the full filtered slice. "
        "Every row is synthetic."
    )
    st.download_button(
        "Download filtered rows as CSV",
        data=_display_frame(view).to_csv(index=False).encode("utf-8"),
        file_name="ctaid-poc-filtered-synthetic.csv",
        mime="text/csv",
    )


def _statistics_view(frame: pd.DataFrame) -> None:
    stats = describe(frame)
    features = feature_columns(frame)
    selected = st.selectbox("Feature for the summary card", options=features, index=0)
    row = stats[stats["feature"] == selected].iloc[0]
    cols = st.columns(5)
    metrics = [
        ("Rows", f"{int(row['count']):,}"),
        ("Mean", f"{row['mean']:.3f}"),
        ("Std dev", f"{row['std']:.3f}"),
        ("Median", f"{row['median']:.3f}"),
        ("Range", f"{row['min']:.3f} to {row['max']:.3f}"),
    ]
    for column, (label, value) in zip(cols, metrics, strict=True):
        with column:
            stat(label, value, truncate(selected, 30))

    st.dataframe(stats, width="stretch", height=430, hide_index=True)
    st.caption("Descriptive statistics are computed live from the loaded synthetic table.")


def _distribution_view(frame: pd.DataFrame) -> None:
    features = feature_columns(frame)
    default = features.index("GLCM_Contrast[=Variance]") if "GLCM_Contrast[=Variance]" in features else 0
    feature = st.selectbox("Feature", options=features, index=default)
    bins = st.slider("Histogram bins", min_value=12, max_value=80, value=40, step=4)

    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(charts.distribution(frame, feature, bins=bins), width="stretch", theme=None)
    with right:
        st.plotly_chart(charts.box_by_class(frame, feature), width="stretch", theme=None)

    st.caption(
        "Histograms overlay each pattern class. Hover for row counts. The box plot shows "
        "spread per class, which is how the generator gives each pattern a distinct shape."
    )


def _relationship_view(frame: pd.DataFrame) -> None:
    features = feature_columns(frame)
    left, right = st.columns(2)
    index_x = features.index("CONVENTIONAL_mean") if "CONVENTIONAL_mean" in features else 0
    index_y = features.index("CONVENTIONAL_max") if "CONVENTIONAL_max" in features else min(1, len(features) - 1)
    with left:
        x_feature = st.selectbox("Horizontal axis", options=features, index=index_x)
    with right:
        y_feature = st.selectbox("Vertical axis", options=features, index=index_y)
    if x_feature == y_feature:
        st.warning("Pick two different features to see a relationship.")
        return
    st.plotly_chart(charts.scatter(frame, x_feature, y_feature), width="stretch", theme=None)


def _correlation_view(frame: pd.DataFrame) -> None:
    method = st.radio(
        "Correlation method",
        options=["pearson", "spearman"],
        horizontal=True,
        help="Pearson for linear relationships, Spearman for rank order.",
    )
    matrix = correlation_matrix(frame, method=method)
    pairs = top_correlations(matrix, limit=12)
    st.plotly_chart(charts.correlation_bars(pairs), width="stretch", theme=None)
    with st.expander("Full correlation heatmap (top 24 features by absolute load)", expanded=False):
        st.plotly_chart(charts.correlation_heatmap(matrix), width="stretch", theme=None)
    st.dataframe(
        pd.DataFrame(pairs, columns=["feature a", "feature b", "r"]).round(3),
        width="stretch",
        hide_index=True,
        height=300,
    )


def render() -> None:
    frame = dataset()
    info = current_summary()

    page_header(
        "Data Exploration",
        "Synthetic feature table",
        "Browse the generated table, compute statistics, compare distributions per pattern "
        "class and inspect feature correlations. All numbers come from the CSV in data/.",
    )

    selected = _filter_frame(frame)
    if selected.empty:
        footer()
        return

    cols = st.columns(4)
    with cols[0]:
        stat("Rows in view", f"{len(selected):,}", f"of {info['rows']:,} total")
    with cols[1]:
        stat("Records", f"{info['records']:,}", "3 vessels each")
    with cols[2]:
        stat("Feature columns", f"{info['feature_count']}", "numeric, no identifiers")
    with cols[3]:
        counts = selected[TARGET_COLUMN].value_counts()
        stat("Largest class", class_label(int(counts.index[0])), f"{int(counts.iloc[0]):,} rows")

    tabs = st.tabs(["Table", "Statistics", "Distributions", "Relationships", "Correlations"])
    with tabs[0]:
        section("Raw rows", "Synthetic records with sequential Record IDs.")
        _table_view(selected)
    with tabs[1]:
        section("Descriptive statistics", "Mean, spread and range per feature.")
        _statistics_view(selected)
    with tabs[2]:
        section("Distributions", "Pick a feature, compare its shape across classes.")
        _distribution_view(selected)
    with tabs[3]:
        section("Relationships", "Two features at a time, coloured by class.")
        _relationship_view(selected)
    with tabs[4]:
        section("Correlations", "Strongest absolute feature pairs first, then the full heatmap.")
        _correlation_view(selected)

    st.markdown(
        f'<p style="color:{MUTED};font-size:0.86rem;">Charts rendered with Plotly using the '
        "project dark palette. No default Streamlit colours are used.</p>",
        unsafe_allow_html=True,
    )
    footer()
