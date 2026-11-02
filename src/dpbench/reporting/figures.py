"""Shared Plotly styling: white background, readable log axes, ε ticks.

The figures have no title of their own (the dashboard puts a heading above each one).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from dpbench.reporting.style import FONT, GRID_COLOR, eps_text


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
