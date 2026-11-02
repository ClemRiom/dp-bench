"""Tables and automatically generated findings shared by the HTML report and the dashboard."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from dpbench.experiments.aggregate import format_mean_std
from dpbench.reporting.style import accountant_name, column_label, display_name, eps_text, lower_is_better

UNCERTIFIED_MARK = " †"


def certified(summary: pd.DataFrame) -> pd.Series:
    """True for the rows whose target ε is certified ("real ε").

    A mechanism may report, besides the ε claimed by its paper (``epsilon_spent``), the ε
    certified in the worst case (``epsilon_worst_case``, e.g. GeoDP). A row is certified
    when there is no such value or when it does not exceed the target.
    """
    if summary.empty or "epsilon_worst_case_mean" not in summary:
        return pd.Series(True, index=summary.index)
    worst = summary["epsilon_worst_case_mean"]
    return worst.isna() | (worst <= summary["epsilon_target"] * (1 + 1e-6))


def real_epsilon(summary: pd.DataFrame) -> pd.Series:
    """The certified ε of every row: the worst-case ε when the mechanism reports one, else the ε spent."""
    spent = summary.get("epsilon_spent_mean", pd.Series(np.nan, index=summary.index))
    if "epsilon_worst_case_mean" in summary:
        return summary["epsilon_worst_case_mean"].fillna(spent)
    return spent


def leaderboard(
    summary: pd.DataFrame, column: str = "test_accuracy", digits: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target ε in rows, mechanisms in columns, 'mean ± std' in cells; and a mask of the best cell of each row.

    The non-private baseline gets its own first row (ε = ∞). The best cell is chosen among
    the certified rows only; the others are marked with a dagger (their ε is only claimed).
    """
    if summary.empty or f"{column}_mean" not in summary:
        return pd.DataFrame(), pd.DataFrame()
    names = {m: display_name(m, n) for m, n in zip(summary["mechanism"], summary["mechanism_name"])}
    mechanisms = list(dict.fromkeys(summary["mechanism"]))
    epsilons = sorted(summary["epsilon_target"].unique(), key=lambda e: (math.isfinite(e), e))
    values = pd.DataFrame(index=[eps_text(e) for e in epsilons], columns=[names[m] for m in mechanisms], dtype=object)
    means = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    is_certified = certified(summary)
    for index, row in summary.iterrows():
        r, c = eps_text(row["epsilon_target"]), names[row["mechanism"]]
        mark = "" if is_certified[index] else UNCERTIFIED_MARK
        values.loc[r, c] = format_mean_std(row[f"{column}_mean"], row.get(f"{column}_std", np.nan), digits) + mark
        if is_certified[index]:
            means.loc[r, c] = row[f"{column}_mean"]
    values = values.fillna("")
    best = pd.DataFrame(False, index=values.index, columns=values.columns)
    for r in means.index:
        row = means.loc[r].dropna()
        if r == "∞" or row.empty:
            continue
        target = row.min() if lower_is_better(column) else row.max()
        best.loc[r] = means.loc[r] == target
    values.index.name = "ε"
    return values, best


def summary_table(summary: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> pd.DataFrame:
    """The summary with display names and 'mean ± std' strings, for the given metric columns."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, r in summary.iterrows():
        row = {
            "Mechanism": display_name(r["mechanism"], r["mechanism_name"]),
            "Target ε": eps_text(r["epsilon_target"]),
            "Seeds": int(r["n_seeds"]),
        }
        for column in columns:
            if f"{column}_mean" in r:
                row[column_label(column)] = format_mean_std(r[f"{column}_mean"], r.get(f"{column}_std"), digits)
        rows.append(row)
    return pd.DataFrame(rows)


def _best(summary: pd.DataFrame, epsilon: float, column: str) -> pd.Series | None:
    """The best private mechanism at this ε among those whose ε is certified."""
    rows = summary[
        summary["private"].astype(bool) & np.isclose(summary["epsilon_target"], epsilon) & certified(summary)
    ]
    rows = rows.dropna(subset=[f"{column}_mean"])
    if rows.empty:
        return None
    index = rows[f"{column}_mean"].idxmin() if lower_is_better(column) else rows[f"{column}_mean"].idxmax()
    return rows.loc[index]


def key_findings(summary: pd.DataFrame, results: pd.DataFrame, column: str = "test_accuracy") -> list[str]:
    """A few factual sentences computed from the results (no interpretation)."""
    findings: list[str] = []
    if summary.empty or f"{column}_mean" not in summary:
        return findings
    label = column_label(column).lower()
    baseline = summary[~summary["private"].astype(bool)]
    base_value = float(baseline[f"{column}_mean"].iloc[0]) if not baseline.empty else math.nan
    if not baseline.empty:
        findings.append(
            f"Non-private baseline: {label} {format_mean_std(base_value, baseline[f'{column}_std'].iloc[0])}."
        )
    epsilons = sorted(e for e in summary["epsilon_target"].unique() if math.isfinite(e))
    for epsilon in dict.fromkeys([epsilons[0], epsilons[-1]] if epsilons else []):
        best = _best(summary, epsilon, column)
        if best is None:
            continue
        text = (
            f"At ε = {eps_text(epsilon)}, the best private mechanism (with a certified ε) is "
            f"{display_name(best['mechanism'], best['mechanism_name'])} with {label} {best[f'{column}_mean']:.3f}"
        )
        if math.isfinite(base_value):
            text += f" ({100 * (best[f'{column}_mean'] - base_value):+.1f} points vs non-private)"
        findings.append(text + ".")
    if not results.empty and "epsilon_spent" in results:
        private = results[(results["status"] == "ok") & results["private"].astype(bool)]
        if not private.empty:
            within = (private["epsilon_spent"] <= private["epsilon_target"] * (1 + 1e-9)).all()
            findings.append(
                f"Accounting check: ε spent ≤ target ε in all {len(private)} private trainings."
                if within
                else "Accounting check FAILED: some trainings spent more than their target ε."
            )
    uncertified = summary[~certified(summary)].sort_values("epsilon_target")
    for _, row in uncertified.drop_duplicates("mechanism").iterrows():
        findings.append(
            f"{display_name(row['mechanism'], row['mechanism_name'])} at ε = {eps_text(row['epsilon_target'])} reaches "
            f"{label} {row[f'{column}_mean']:.3f}, but only with the ε claimed by its paper "
            f"({row['epsilon_spent_mean']:.3g}): the ε certified in the worst case is "
            f"{row['epsilon_worst_case_mean']:.3g}, so it is not ranked (marked † in the tables)."
        )
    names = set(summary["mechanism_name"])
    if {"dpsgd", "dpsgd_opacus"} <= names:
        ours = summary[summary["mechanism_name"] == "dpsgd"].set_index("epsilon_target")[f"{column}_mean"]
        theirs = summary[summary["mechanism_name"] == "dpsgd_opacus"].set_index("epsilon_target")[f"{column}_mean"]
        common = ours.index.intersection(theirs.index)
        if len(common):
            gap = float(np.mean(np.abs(ours[common] - theirs[common])))
            findings.append(
                f"Cross-check: the in-house DP-SGD and Opacus differ by {100 * gap:.1f} points of {label} "
                "on average over the budgets (same noise, different random draws)."
            )
    if "time_per_epoch_s_mean" in summary and not baseline.empty:
        reference = float(baseline["time_per_epoch_s_mean"].iloc[0])
        dpsgd = summary[summary["mechanism_name"] == "dpsgd"]
        if reference > 0 and not dpsgd.empty:
            findings.append(
                f"Cost: an epoch of DP-SGD takes {dpsgd['time_per_epoch_s_mean'].mean() / reference:.1f}× "
                "the time of a non-private epoch (per-example gradients)."
            )
    return findings


def calibration_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Accountant, noise multiplier and ε spent of every private (mechanism, target ε)."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, r in summary[summary["private"].astype(bool)].iterrows():
        row = {
            "Mechanism": display_name(r["mechanism"], r["mechanism_name"]),
            "Target ε": eps_text(r["epsilon_target"]),
            "Accountant": accountant_name(str(r.get("accountant", "") or "")),
            "Noise multiplier σ": f"{r['noise_multiplier_mean']:.4f}" if "noise_multiplier_mean" in r else "",
            "Sampling rate q": f"{r['sample_rate_mean']:.4g}" if "sample_rate_mean" in r else "",
            "Steps": f"{r['steps_mean']:.0f}" if "steps_mean" in r else "",
            "ε spent": f"{r['epsilon_spent_mean']:.4f}" if "epsilon_spent_mean" in r else "",
        }
        if "epsilon_opacus_mean" in r and np.isfinite(r["epsilon_opacus_mean"]):
            row["ε (Opacus' accountant)"] = f"{r['epsilon_opacus_mean']:.4f}"
        elif "epsilon_opacus_mean" in summary:
            row["ε (Opacus' accountant)"] = ""
        if "epsilon_worst_case_mean" in summary:
            value = r.get("epsilon_worst_case_mean", math.nan)
            row["ε certified (worst case)"] = f"{value:.4g}" if pd.notna(value) else ""
        rows.append(row)
    return pd.DataFrame(rows)
