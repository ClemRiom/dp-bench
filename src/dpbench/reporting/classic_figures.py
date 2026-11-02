"""Plotly figures of a classic-DP benchmark (queries on a table).

Mechanisms keep the colours of the rest of DP-Bench (one fixed colour per mechanism,
in the order of the benchmark). Errors span several orders of magnitude, so both axes
are logarithmic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dpbench.reporting.figures import style
from dpbench.reporting.style import PALETTE, eps_text

METRIC_LABELS = {
    "abs_error": "Absolute error",
    "rel_error": "Relative error",
    "max_error": "Largest error",
    "correct": "Fraction of correct answers",
    "time_ms": "Time per release (ms)",
}


DASHES = ("solid", "dash", "dot")


def mechanism_styles(mechanisms: list[str]) -> dict[str, tuple[str, str]]:
    """(colour, dash) of each mechanism, in a fixed order: the 10th mechanism reuses the first colour, dashed."""
    order = list(dict.fromkeys(mechanisms))
    return {m: (PALETTE[i % len(PALETTE)], DASHES[(i // len(PALETTE)) % len(DASHES)]) for i, m in enumerate(order)}


def compact(value: float) -> str:
    """A short label for an error: 0.014, 0.12, 2.5, 1234, 4.4e-5 (at most 6 characters for most values)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if value == 0 or not math.isfinite(value):
        return f"{value:g}"
    size = abs(value)
    if 0.01 <= size < 10_000:
        return f"{value:.2g}" if size < 100 else f"{value:.0f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _decade_ticks(fig: go.Figure) -> None:
    """Powers of ten only on a log y axis that spans more than two decades (1-2-5 ticks get crowded)."""
    values = np.concatenate([np.asarray(t.y, dtype=float) for t in fig.data if t.y is not None] or [np.array([])])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0 or values.max() / values.min() < 100:
        return
    low, high = math.floor(math.log10(values.min())), math.ceil(math.log10(values.max()))
    exponents = list(range(low, high + 1))
    fig.update_yaxes(tickvals=[10.0**k for k in exponents], ticktext=[power_of_ten(k) for k in exponents])


SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def power_of_ten(exponent: int) -> str:
    """0.01, 0.1, 1, 10, 100, 1000 written out; 10⁻⁵ or 10⁶ beyond."""
    return f"{10.0**exponent:g}" if -2 <= exponent <= 3 else "10" + str(exponent).translate(SUPERSCRIPTS)


def error_vs_epsilon(summary: pd.DataFrame, query: str, metric: str = "rel_error") -> go.Figure:
    """Mean error (± standard deviation over the repetitions) of every mechanism against ε, for one query."""
    fig = go.Figure()
    data = summary[summary["query"] == query]
    column = f"{metric}_mean"
    if data.empty or column not in data:
        return style(fig, "Privacy budget ε (log scale)", METRIC_LABELS.get(metric, metric))
    # One style per mechanism of this query, in the order of the benchmark (so at most 9 plain colours usually).
    styles = mechanism_styles([m for m in dict.fromkeys(summary["mechanism"]) if m in set(data["mechanism"])])
    log_y = metric != "correct"
    for mechanism, group in data.groupby("mechanism", sort=False):
        group = group.sort_values("epsilon")
        y = group[column].to_numpy(dtype=float)
        std = group.get(f"{metric}_std", pd.Series(0.0, index=group.index)).fillna(0.0).to_numpy(dtype=float)
        if log_y:
            y = np.where(y > 0, y, np.nan)
        fig.add_trace(
            go.Scatter(
                x=group["epsilon"],
                y=y,
                mode="lines+markers",
                name=mechanism,
                line=dict(color=styles[mechanism][0], dash=styles[mechanism][1], width=2),
                marker=dict(size=8, color=styles[mechanism][0], line=dict(color="white", width=2)),
                error_y=dict(type="data", array=std, visible=not log_y, thickness=1.2, width=3),
                hovertemplate=f"<b>{mechanism}</b><br>ε = %{{x:g}}<br>{METRIC_LABELS.get(metric, metric)} = %{{y:.4g}}<extra></extra>",
            )
        )
    epsilons = sorted(data["epsilon"].unique())
    title = METRIC_LABELS.get(metric, metric) + (" (log scale)" if log_y else "")
    fig = style(fig, "Privacy budget ε (log scale)", title, log_y=log_y, epsilons=epsilons, height=460)
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02), margin=dict(l=72, r=24, t=24, b=56)
    )
    if log_y:
        _decade_ticks(fig)
    else:
        fig.update_yaxes(range=[-0.03, 1.03] if metric == "correct" else None)
    return fig


def error_heatmap(summary: pd.DataFrame, epsilon: float, metric: str = "rel_error") -> go.Figure:
    """Queries in rows, mechanisms in columns: the mean error at one ε (blank = the mechanism cannot answer).

    The colour compares the mechanisms of a row: lightest for the best one, darker for an
    error 10 times larger, darkest for 100 times larger or more.
    """
    data = summary[np.isclose(summary["epsilon"], epsilon)]
    column = f"{metric}_mean"  # the mean, like the curves (a median is often exactly 0 for integer noise)
    queries = list(dict.fromkeys(summary["query"]))
    mechanisms = list(dict.fromkeys(summary["mechanism"]))
    values = np.full((len(queries), len(mechanisms)), np.nan)
    for _, row in data.iterrows():
        values[queries.index(row["query"]), mechanisms.index(row["mechanism"])] = row[column]
    best = np.nanmin(np.where(np.isnan(values), np.inf, values), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(values == best[:, None], 1.0, values / best[:, None])  # 1 for the best, even if it is 0
        shown = np.clip(np.log10(ratio), 0.0, 2.0)
    text = [["" if math.isnan(v) else compact(v) for v in line] for line in values]
    for i, line in enumerate(values):
        for j, v in enumerate(line):
            if not math.isnan(v) and v == best[i] and len(np.flatnonzero(~np.isnan(line))) > 1:
                text[i][j] = f"<b>{text[i][j]}</b>"
    fig = go.Figure(
        go.Heatmap(
            z=shown,
            x=mechanisms,
            y=queries,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[[0.0, "#eef4fc"], [0.5, "#86b6e8"], [1.0, "#1c5aa6"]],
            zmin=0.0,
            zmax=2.0,
            showscale=False,
            xgap=2,
            ygap=2,
            hoverongaps=False,
            customdata=np.dstack([values, ratio]),
            hovertemplate="<b>%{y}</b><br>%{x}<br>mean "
            + METRIC_LABELS.get(metric, metric).lower()
            + " = %{customdata[0]:.4g}<br>%{customdata[1]:.3g} × the best of the row<extra></extra>",
        )
    )
    fig = style(fig, "", "", log_x=False, height=max(260, 70 + 44 * len(queries)))
    fig.update_layout(margin=dict(l=210, r=24, t=24, b=120))
    fig.update_xaxes(showline=False, ticks="", tickangle=-30, showgrid=False, side="bottom")
    fig.update_yaxes(showline=False, ticks="", autorange="reversed", showgrid=False)
    fig.add_annotation(
        text=f"mean {METRIC_LABELS.get(metric, metric).lower()} at ε = {eps_text(epsilon)} · bold = best of the row · "
        "darker = 10×, 100× larger than the best",
        xref="paper",
        yref="paper",
        x=0,
        y=1.0,
        yanchor="bottom",
        showarrow=False,
        font=dict(size=12, color="#5d5d5a"),
    )
    return fig


def error_vs_records(summary: pd.DataFrame, query: str, epsilon: float, metric: str = "rel_error") -> go.Figure:
    """Mean error of every mechanism against the number of records of each database, for one query and one ε."""
    fig = go.Figure()
    data = summary[(summary["query"] == query) & np.isclose(summary["epsilon"], epsilon)]
    column = f"{metric}_mean"
    if data.empty or column not in data:
        return style(fig, "Number of records (log scale)", METRIC_LABELS.get(metric, metric))
    records = data.groupby("table")["records"].first().sort_values()
    styles = mechanism_styles([m for m in dict.fromkeys(summary["mechanism"]) if m in set(data["mechanism"])])
    for mechanism, (color, dash) in styles.items():
        rows = data[data["mechanism"] == mechanism].set_index("table").reindex(records.index)
        y = rows[column].to_numpy(dtype=float)
        y = np.where(y > 0, y, np.nan)
        fig.add_trace(
            go.Scatter(
                x=records.to_numpy(dtype=float),
                y=y,
                mode="lines+markers",
                name=mechanism,
                line=dict(color=color, dash=dash, width=2),
                marker=dict(size=9, color=color, line=dict(color="white", width=2)),
                customdata=records.index,
                hovertemplate=f"<b>{mechanism}</b><br>%{{customdata}} (%{{x:,}} records)<br>"
                f"{METRIC_LABELS.get(metric, metric)} = %{{y:.4g}}<extra></extra>",
            )
        )
    fig = style(fig, "Number of records (log scale)", METRIC_LABELS.get(metric, metric) + " (log scale)", log_y=True)
    fig.update_xaxes(
        tickvals=records.to_numpy(dtype=float), ticktext=[f"{t}<br>{n:,}" for t, n in records.items()], range=None
    )
    fig.update_layout(
        height=460,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=72, r=24, t=24, b=72),
    )
    _decade_ticks(fig)
    return fig
