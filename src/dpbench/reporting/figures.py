"""Plotly figures of a run. Every function takes the run's data frames and returns a ``go.Figure``.

The figures have no title of their own (the report and the dashboard put a heading above
each one). Mechanisms keep the same colour in every figure; the non-private baseline is a
dark grey dashed line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from dpbench.reporting.style import (
    BASELINE_COLOR,
    BOUND_COLOR,
    FONT,
    GRID_COLOR,
    PALETTE,
    accountant_name,
    color_map,
    column_label,
    display_name,
    eps_text,
)


def nice_log_ticks(low: float, high: float) -> list[float]:
    """Readable tick values (1-2-5 sequence, or powers of ten on wide ranges) for a log axis."""
    if not (np.isfinite(low) and np.isfinite(high)) or low <= 0 or high <= 0:
        return []
    low, high = min(low, high), max(low, high)
    decades = math.log10(high) - math.log10(low)
    mantissas = (1, 2, 3, 5) if decades <= 1.2 else (1, 2, 5) if decades <= 3.5 else (1,)
    ticks = [
        m * 10.0**k
        for k in range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 2)
        for m in mantissas
        if low / 1.25 <= m * 10.0**k <= high * 1.25
    ]
    return [float(f"{t:.6g}") for t in ticks]


def _trace_values(fig: go.Figure, axis: str) -> np.ndarray:
    values = []
    for trace in fig.data:
        data = getattr(trace, axis, None)
        if data is None:
            continue
        array = np.asarray(data, dtype=object)
        try:
            numeric = array.astype(float)
        except (TypeError, ValueError):
            continue
        values.append(numeric[np.isfinite(numeric) & (numeric > 0)])
    return np.concatenate(values) if values else np.array([])


def _log_axis(fig: go.Figure, axis: str) -> None:
    values = _trace_values(fig, axis)
    if values.size:
        ticks = nice_log_ticks(float(values.min()), float(values.max()))
        if ticks:
            update = fig.update_xaxes if axis == "x" else fig.update_yaxes
            update(tickvals=ticks, ticktext=[f"{t:g}" for t in ticks])


def style(
    fig: go.Figure,
    x_title: str,
    y_title: str,
    *,
    log_x: bool = True,
    log_y: bool = False,
    height: int = 430,
    epsilons: Sequence[float] | None = None,
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=13, color="#1f1f1f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=64, r=24, t=48, b=56),
        height=height,
        hovermode="closest",
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(
        title_text=x_title,
        showline=True,
        linecolor="#9a9a9a",
        ticks="outside",
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=y_title,
        showline=True,
        linecolor="#9a9a9a",
        ticks="outside",
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    if log_x:
        fig.update_xaxes(type="log")
        if epsilons:
            fig.update_xaxes(tickvals=list(epsilons), ticktext=[eps_text(e) for e in epsilons])
        else:
            _log_axis(fig, "x")
    if log_y:
        fig.update_yaxes(type="log")
        _log_axis(fig, "y")
    return fig


def _colors(summary: pd.DataFrame) -> dict[str, str]:
    baselines = summary.loc[~summary["private"].astype(bool), "mechanism"] if not summary.empty else []
    return color_map(summary["mechanism"] if not summary.empty else [], baselines)


def _private(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["private"].astype(bool)]


def _baselines(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[~summary["private"].astype(bool)]


def _epsilons(summary: pd.DataFrame) -> list[float]:
    return sorted(e for e in summary["epsilon_target"].unique() if math.isfinite(e))


def _mechanism_traces(fig: go.Figure, summary: pd.DataFrame, column: str, colors: dict[str, str]) -> None:
    label = column_label(column)
    for mechanism, group in _private(summary).groupby("mechanism", sort=False):
        group = group.sort_values("epsilon_target")
        if f"{column}_mean" not in group or group[f"{column}_mean"].isna().all():
            continue
        name = display_name(mechanism, group["mechanism_name"].iloc[0])
        std = group[f"{column}_std"].fillna(0.0).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=group["epsilon_target"],
                y=group[f"{column}_mean"],
                error_y=dict(type="data", array=std, visible=True, thickness=1.2, width=4),
                mode="lines+markers",
                name=name,
                line=dict(color=colors[mechanism], width=2.5),
                marker=dict(size=8),
                customdata=np.stack([std, group["n_seeds"].to_numpy()], axis=1),
                hovertemplate=(
                    f"<b>{name}</b><br>ε = %{{x:g}}<br>{label} = %{{y:.4f}} ± %{{customdata[0]:.4f}}"
                    "<br>seeds: %{customdata[1]}<extra></extra>"
                ),
            )
        )


def _baseline_lines(fig: go.Figure, summary: pd.DataFrame, column: str) -> None:
    for _, row in _baselines(summary).iterrows():
        value = row.get(f"{column}_mean")
        if value is None or not np.isfinite(value):
            continue
        name = display_name(row["mechanism"], row["mechanism_name"])
        fig.add_hline(
            y=value,
            line=dict(color=BASELINE_COLOR, dash="dash", width=2),
            annotation_text=f"{name}: {value:.3f}",
            annotation_position="top left",
            annotation_font=dict(color=BASELINE_COLOR, size=12),
        )


def privacy_utility(
    summary: pd.DataFrame, column: str = "test_accuracy", reference: tuple[str, float] | None = None
) -> go.Figure:
    """A metric against the privacy budget, one line per private mechanism (mean ± std over seeds)."""
    fig = go.Figure()
    if summary.empty:
        return style(fig, "Privacy budget ε", column_label(column))
    _mechanism_traces(fig, summary, column, _colors(summary))
    _baseline_lines(fig, summary, column)
    if reference is not None and np.isfinite(reference[1]):
        fig.add_hline(
            y=reference[1],
            line=dict(color="#8a8a8a", dash="dot", width=1.5),
            annotation_text=f"{reference[0]}: {reference[1]:.3f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#6a6a6a", size=12),
        )
    return style(fig, "Privacy budget ε (log scale)", column_label(column), epsilons=_epsilons(summary))


def noise_vs_epsilon(summary: pd.DataFrame) -> go.Figure:
    """Calibrated noise multiplier against the target ε."""
    fig = go.Figure()
    if summary.empty or "noise_multiplier_mean" not in summary:
        return style(fig, "Privacy budget ε", "Noise multiplier σ")
    _mechanism_traces(fig, summary, "noise_multiplier", _colors(summary))
    return style(fig, "Target ε (log scale)", "Noise multiplier σ (log scale)", log_y=True, epsilons=_epsilons(summary))


def epsilon_check(summary: pd.DataFrame) -> go.Figure:
    """ε spent according to the accountant against the target: every point must lie on or below the diagonal."""
    fig = go.Figure()
    private = _private(summary)
    if private.empty:
        return style(fig, "Target ε", "ε spent")
    epsilons = _epsilons(summary)
    fig.add_trace(
        go.Scatter(
            x=epsilons,
            y=epsilons,
            mode="lines",
            name="ε spent = target",
            line=dict(color=BOUND_COLOR, dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    colors = _colors(summary)
    _mechanism_traces(fig, summary, "epsilon_spent", colors)
    # Mechanisms that also report the ε certified in the worst case (GeoDP): one cross per budget.
    worst = private.iloc[:0]
    if "epsilon_worst_case_mean" in private:
        worst = private.dropna(subset=["epsilon_worst_case_mean"])
        worst = worst[worst["epsilon_worst_case_mean"] > worst["epsilon_target"] * (1 + 1e-6)]
    for mechanism, group in worst.groupby("mechanism", sort=False):
        group = group.sort_values("epsilon_target")
        name = display_name(mechanism, group["mechanism_name"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=group["epsilon_target"],
                y=group["epsilon_worst_case_mean"],
                mode="lines+markers",
                name=f"{name}: certified ε",
                line=dict(color=colors.get(mechanism, "#555"), dash="dot", width=1.5),
                marker=dict(symbol="x", size=10, color=colors.get(mechanism, "#555")),
                hovertemplate=f"<b>{name}</b><br>target ε = %{{x:g}}<br>certified ε = %{{y:.3g}}<extra></extra>",
            )
        )
    fig = style(fig, "Target ε (log scale)", "ε spent (log scale)", log_y=True, epsilons=epsilons)
    if worst.empty:
        fig.update_yaxes(tickvals=epsilons, ticktext=[eps_text(e) for e in epsilons])
    return fig


def epsilon_over_training(history: pd.DataFrame, mechanism: str) -> go.Figure:
    """ε spent so far against the epoch, one line per target ε (the 'ε versus steps' curve)."""
    fig = go.Figure()
    if history.empty or "epsilon" not in history:
        return style(fig, "Epoch", "ε spent so far", log_x=False)
    data = history[(history["mechanism"] == mechanism) & np.isfinite(history["epsilon"])]
    targets = sorted(data["epsilon_target"].unique())
    colors = sample_colorscale("Blues", [0.35 + 0.65 * i / max(len(targets) - 1, 1) for i in range(len(targets))])
    for color, target in zip(colors, targets):
        curve = data[data["epsilon_target"] == target].groupby("epoch")["epsilon"].mean()
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve.values,
                mode="lines+markers",
                name=f"target ε = {eps_text(target)}",
                line=dict(color=color, width=2.2),
                marker=dict(size=6),
                hovertemplate="epoch %{x}<br>ε spent = %{y:.3f}<extra></extra>",
            )
        )
    return style(fig, "Epoch", "ε spent so far (log scale)", log_x=False, log_y=True)


def accountant_curves(frame: pd.DataFrame) -> go.Figure:
    """ε against the number of epochs for several accountants (columns: accountant, epoch, epsilon)."""
    fig = go.Figure()
    for i, (accountant, group) in enumerate(frame.groupby("accountant", sort=False)):
        dash = "dot" if group.get("approximate", pd.Series([False])).iloc[0] else "solid"  # GDP: an approximation
        fig.add_trace(
            go.Scatter(
                x=group["epoch"],
                y=group["epsilon"],
                mode="lines+markers",
                name=accountant_name(accountant),
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.3, dash=dash),
                marker=dict(size=5),
                hovertemplate=f"<b>{accountant_name(accountant)}</b><br>epoch %{{x}}<br>ε = %{{y:.4f}}<extra></extra>",
            )
        )
    return style(fig, "Epochs", "ε (log scale)", log_x=False, log_y=True)
