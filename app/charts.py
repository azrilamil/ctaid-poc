"""Plotly chart builders for the CTAID POC demo.

All charts share one dark palette so the app looks like a single product rather
than a pile of library defaults.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.schema import CLASS_LABELS, TARGET_COLUMN

PALETTE = ["#5AC8FA", "#FF7A59", "#9B8CFF", "#43D9A3"]
ACCENT = "#5AC8FA"
GRID = "rgba(255,255,255,0.08)"
PAPER = "rgba(0,0,0,0)"
FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", color="#D7E1F0", size=13)

BASE_LAYOUT = dict(
    paper_bgcolor=PAPER,
    plot_bgcolor=PAPER,
    font=FONT,
    margin=dict(l=10, r=10, t=48, b=10),
    hoverlabel=dict(bgcolor="#101B2D", bordercolor="#2A3A57", font=dict(color="#E7EEF9")),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
)

AXIS = dict(
    gridcolor=GRID,
    zerolinecolor=GRID,
    linecolor=GRID,
    tickfont=dict(color="#9FB0C9", size=12),
    title=dict(font=dict(color="#9FB0C9", size=12)),
)


def _apply(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, height=height)
    if title:
        fig.update_layout(
            title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15, color="#E7EEF9"))
        )
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


def class_color(class_id: int) -> str:
    return PALETTE[int(class_id) % len(PALETTE)]


def class_label(class_id: int) -> str:
    return CLASS_LABELS.get(int(class_id), f"Class {int(class_id)}")


def class_balance(frame: pd.DataFrame) -> go.Figure:
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    fig = go.Figure(
        go.Bar(
            x=[class_label(int(k)) for k in counts.index],
            y=counts.to_numpy(),
            marker=dict(color=[class_color(int(k)) for k in counts.index], line=dict(width=0)),
            text=counts.to_numpy(),
            textposition="outside",
            textfont=dict(color="#C9D6E8"),
            hovertemplate="%{x}<br>%{y} rows<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.45, showlegend=False)
    fig.update_yaxes(title="Rows", rangemode="tozero")
    return _apply(fig, "Rows per pattern class", height=300)


def distribution(frame: pd.DataFrame, feature: str, bins: int = 40, show_box: bool = True) -> go.Figure:
    fig = go.Figure()
    for class_id in sorted(frame[TARGET_COLUMN].unique()):
        subset = frame[frame[TARGET_COLUMN] == class_id][feature]
        fig.add_trace(
            go.Histogram(
                x=subset,
                name=class_label(int(class_id)),
                nbinsx=bins,
                opacity=0.62,
                marker=dict(color=class_color(int(class_id)), line=dict(width=0)),
                hovertemplate="%{x}<br>%{y} rows<extra>" + class_label(int(class_id)) + "</extra>",
            )
        )
    fig.update_layout(barmode="overlay", bargap=0.02)
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title="Rows")
    if show_box:
        fig.update_layout(margin=dict(l=10, r=10, t=48, b=60))
    return _apply(fig, f"Distribution of {feature}", height=430)


def box_by_class(frame: pd.DataFrame, feature: str) -> go.Figure:
    fig = go.Figure()
    for class_id in sorted(frame[TARGET_COLUMN].unique()):
        subset = frame[frame[TARGET_COLUMN] == class_id][feature]
        fig.add_trace(
            go.Box(
                y=subset,
                name=class_label(int(class_id)),
                marker=dict(color=class_color(int(class_id))),
                boxpoints="outliers",
                line=dict(width=1.4),
            )
        )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=feature)
    return _apply(fig, f"{feature} spread per class", height=430)


def scatter(frame: pd.DataFrame, x_feature: str, y_feature: str, max_points: int = 900) -> go.Figure:
    sample = frame if len(frame) <= max_points else frame.sample(max_points, random_state=7)
    fig = go.Figure()
    for class_id in sorted(sample[TARGET_COLUMN].unique()):
        subset = sample[sample[TARGET_COLUMN] == class_id]
        fig.add_trace(
            go.Scattergl(
                x=subset[x_feature],
                y=subset[y_feature],
                mode="markers",
                name=class_label(int(class_id)),
                marker=dict(color=class_color(int(class_id)), size=6, opacity=0.75, line=dict(width=0)),
                hovertemplate=(
                    f"{x_feature}: %{{x}}<br>{y_feature}: %{{y}}<br>"
                    + class_label(int(class_id))
                    + "<extra></extra>"
                ),
            )
        )
    fig.update_xaxes(title=x_feature)
    fig.update_yaxes(title=y_feature)
    return _apply(fig, f"{y_feature} against {x_feature}", height=460)


def correlation_heatmap(matrix: pd.DataFrame, limit: int = 24) -> go.Figure:
    """Heatmap of the most variable features, kept readable."""
    if matrix.shape[0] > limit:
        variance = matrix.abs().sum().sort_values(ascending=False)
        keep = list(variance.index[:limit])
        matrix = matrix.loc[keep, keep]
    labels = [name if len(name) <= 28 else name[:26] + ".." for name in matrix.columns]
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#FF6B6B"],
                [0.5, "#0E1726"],
                [1.0, "#5AC8FA"],
            ],
            zmin=-1,
            zmax=1,
            colorbar=dict(title=dict(text="r", font=dict(color="#9FB0C9")), tickfont=dict(color="#9FB0C9")),
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9, color="#9FB0C9"))
    fig.update_yaxes(tickfont=dict(size=9, color="#9FB0C9"))
    return _apply(fig, "Feature correlation (Pearson)", height=620)


def correlation_bars(pairs: Iterable[tuple[str, str, float]], limit: int = 12) -> go.Figure:
    items = list(pairs)[:limit]
    labels = [f"{left}  /  {right}" for left, right, _value in items]
    values = [value for _left, _right, value in items]
    colors = [ACCENT if value >= 0 else "#FF7A59" for value in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Pearson r", range=[-1, 1])
    fig.update_yaxes(autorange="reversed")
    return _apply(fig, "Strongest correlations", height=max(320, 26 * max(len(items), 1) + 90))


def feature_ranking(pairs: list[tuple[str, float]]) -> go.Figure:
    labels = [name for name, _score in pairs][::-1]
    values = [score for _name, score in pairs][::-1]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=ACCENT, opacity=0.85, line=dict(width=0)),
            hovertemplate="%{y}<br>F = %{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Univariate F score")
    return _apply(fig, "How separable each feature is", height=max(320, 30 * max(len(pairs), 1) + 90))


def confusion(matrix: np.ndarray, labels: list[int], title: str) -> go.Figure:
    names = [class_label(label) for label in labels]
    fig = go.Figure(
        go.Heatmap(
            z=np.asarray(matrix),
            x=names,
            y=names,
            colorscale=[[0.0, "#0E1726"], [1.0, ACCENT]],
            showscale=False,
            text=np.asarray(matrix),
            texttemplate="%{text}",
            textfont=dict(color="#F2F6FC", size=14),
            hovertemplate="actual %{y}<br>predicted %{x}<br>%{z} rows<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return _apply(fig, title, height=430)


def roc(curves: dict[int, dict], title: str) -> go.Figure:
    fig = go.Figure()
    for position, (class_id, data) in enumerate(sorted(curves.items())):
        fig.add_trace(
            go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                mode="lines",
                name=f"{class_label(int(class_id))} (AUC {data['auc']:.2f})",
                line=dict(width=2.4, color=PALETTE[position % len(PALETTE)]),
                hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line=dict(width=1.2, color="#6B7B95", dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _apply(fig, title, height=430)


def metric_bars(per_model: dict[str, dict[str, float]], metrics: list[str]) -> go.Figure:
    fig = go.Figure()
    for position, (model_label, values) in enumerate(per_model.items()):
        fig.add_trace(
            go.Bar(
                x=[name.replace("_", " ") for name in metrics],
                y=[values.get(name, 0.0) for name in metrics],
                name=model_label,
                marker=dict(color=PALETTE[position % len(PALETTE)], line=dict(width=0)),
                hovertemplate="%{x}<br>%{y:.3f}<extra>" + model_label + "</extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(title="Score", range=[0, 1.05])
    return _apply(fig, "Metrics by model", height=380)


def importance(pairs: list[tuple[str, float]], title: str) -> go.Figure:
    if not pairs:
        fig = go.Figure()
        fig.add_annotation(
            text="This model does not expose feature importances.",
            showarrow=False,
            font=dict(color="#9FB0C9"),
        )
        return _apply(fig, title, height=300)
    labels = [name for name, _value in pairs][::-1]
    values = [value for _name, value in pairs][::-1]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color="#9B8CFF", opacity=0.9, line=dict(width=0)),
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Relative importance")
    return _apply(fig, title, height=max(320, 28 * len(pairs) + 90))


def probability_bars(probabilities: dict[int, float]) -> go.Figure:
    keys = sorted(probabilities)
    values = [probabilities[key] for key in keys]
    colors = [class_color(key) for key in keys]
    fig = go.Figure(
        go.Bar(
            x=[class_label(key) for key in keys],
            y=values,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{value * 100:.1f}%" for value in values],
            textposition="outside",
            textfont=dict(color="#C9D6E8"),
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False, bargap=0.45)
    fig.update_yaxes(title="Model probability", range=[0, max(1.0, max(values) * 1.25 if values else 1.0)])
    return _apply(fig, "Class probabilities", height=320)


def class_profile(profile: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=profile.to_numpy(),
            x=list(profile.columns),
            y=list(profile.index),
            colorscale=[[0.0, "#0E1726"], [1.0, "#5AC8FA"]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>value %{z:.3f}<extra></extra>",
        )
    )
    return _apply(fig, "Average value of the most separable features", height=420)


def sparkline(series: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            y=series.to_numpy(),
            x=list(range(len(series))),
            mode="lines",
            line=dict(color=ACCENT, width=2),
            fill="tozeroy",
            fillcolor="rgba(90,200,250,0.14)",
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply(fig, None, height=120)
