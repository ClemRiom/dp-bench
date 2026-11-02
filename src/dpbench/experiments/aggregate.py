"""Aggregation of the per-seed results: mean and standard deviation for each (mechanism, epsilon)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

KEYS = ("mechanism", "epsilon_target")
NON_METRICS = {"seed", "epsilon_target", "private"}


def numeric_columns(results: pd.DataFrame) -> list[str]:
    return [
        c
        for c in results.columns
        if c not in NON_METRICS
        and pd.api.types.is_numeric_dtype(results[c])
        and not pd.api.types.is_bool_dtype(results[c])
    ]


def summarize(results: pd.DataFrame, order: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per (mechanism, target epsilon): ``<metric>_mean``, ``<metric>_std`` and ``n_seeds``.

    Failed runs are ignored. The standard deviation uses ddof=1 (it is NaN with one seed).
    """
    if results.empty or "status" not in results:
        return pd.DataFrame()
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    columns = numeric_columns(ok)
    grouped = ok.groupby(list(KEYS), sort=False)
    mean = grouped[columns].mean().add_suffix("_mean")
    std = grouped[columns].std(ddof=1).add_suffix("_std")
    first = {"mechanism_name": ("mechanism_name", "first"), "private": ("private", "first")}
    if "accountant" in ok:
        first["accountant"] = ("accountant", "first")
    info = grouped.agg(**first, n_seeds=("seed", "nunique"))
    summary = pd.concat([info, mean, std], axis=1).reset_index()
    ordered = [c for c in summary.columns if not c.endswith(("_mean", "_std"))]
    for column in columns:
        ordered += [f"{column}_mean", f"{column}_std"]
    summary = summary[ordered]
    rank = {label: i for i, label in enumerate(order or [])}
    summary["_rank"] = summary["mechanism"].map(lambda m: rank.get(m, len(rank)))
    summary = summary.sort_values(["_rank", "epsilon_target"]).drop(columns="_rank").reset_index(drop=True)
    return summary


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
