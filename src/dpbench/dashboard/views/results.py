"""Results page: compare the mechanisms of a run on every axis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from dpbench.dashboard import common
from dpbench.experiments.storage import RESULTS, SUMMARY, Run
from dpbench.reporting import figures
from dpbench.reporting.style import (
    FAIRNESS_COLUMNS,
    UTILITY_COLUMNS,
    column_label,
    display_name,
    eps_text,
)
from dpbench.reporting.tables import (
    calibration_table,
    certified,
    key_findings,
    leaderboard,
    real_epsilon,
    summary_table,
)

BEST_STYLE = "font-weight: 700; color: #0b5bb5"


def _styled_leaderboard(board: pd.DataFrame, best: pd.DataFrame):
    if board.empty:
        return board
    css = pd.DataFrame(np.where(best.to_numpy(dtype=bool), BEST_STYLE, ""), index=best.index, columns=best.columns)
    return board.style.apply(lambda _: css, axis=None)


def _kpis(summary: pd.DataFrame) -> None:
    baseline = summary[~summary["private"].astype(bool)]
    private = summary[summary["private"].astype(bool)]
    if "test_accuracy_mean" not in summary:
        return
    base = float(baseline["test_accuracy_mean"].iloc[0]) if not baseline.empty else math.nan
    columns = st.columns(4)
    columns[0].metric("Non-private accuracy", f"{base:.3f}" if math.isfinite(base) else "-", border=True)
    epsilons = sorted(e for e in private["epsilon_target"].unique())
    ranked = private[certified(private)]  # only the mechanisms whose ε is certified are ranked
    for column, epsilon in zip(columns[1:3], [epsilons[-1], epsilons[0]] if epsilons else []):
        rows = ranked[np.isclose(ranked["epsilon_target"], epsilon)].dropna(subset=["test_accuracy_mean"])
        if rows.empty:
            continue
        best = rows.loc[rows["test_accuracy_mean"].idxmax()]
        delta = f"{100 * (best['test_accuracy_mean'] - base):+.1f} pts vs non-private" if math.isfinite(base) else None
        column.metric(
            f"Best at ε = {eps_text(epsilon)}: {display_name(best['mechanism'], best['mechanism_name'])}",
            f"{best['test_accuracy_mean']:.3f}",
            delta,
            border=True,
        )
    if "epsilon_spent_mean" in private and not private.empty:
        ratio = float((real_epsilon(private) / private["epsilon_target"]).max())
        offenders = sorted(
            {
                display_name(m, n)
                for m, n, ok in zip(private["mechanism"], private["mechanism_name"], certified(private))
                if not ok
            }
        )
        columns[3].metric(
            "Largest certified ε / target ε",
            (f"{ratio:.3f}" if ratio < 1000 else f"{ratio:.2g}") if math.isfinite(ratio) else "-",
            f"not certified: {', '.join(offenders)}" if offenders else None,
            delta_color="inverse",
            border=True,
        )


def _utility(run: Run, summary: pd.DataFrame) -> None:
    options = [c for c in UTILITY_COLUMNS if f"{c}_mean" in summary]
    column = st.segmented_control("Metric", options, default=options[0], format_func=column_label, key="utility_metric")
    column = column or options[0]
    majority = run.dataset.get("majority_class_test_accuracy")
    reference = ("Majority class", float(majority)) if column == "test_accuracy" and majority is not None else None
    common.plot(figures.privacy_utility(summary, column, reference), key="fig_utility")
    board, best = leaderboard(summary, column)
    st.markdown(f"**{column_label(column)} by budget**")
    common.table(_styled_leaderboard(board, best), hide_index=False)
    if not certified(summary).all():
        st.markdown("† ε claimed by the paper but not certified in the worst case: not ranked.")


def _accounting(run: Run, summary: pd.DataFrame) -> None:
    private = [m for m in run.private_mechanisms if m in set(summary["mechanism"])]
    left, right = st.columns(2)
    with left:
        st.markdown("**Calibrated noise multiplier**")
        common.plot(figures.noise_vs_epsilon(summary), key="fig_noise")
    with right:
        st.markdown("**ε spent against the target**")
        common.plot(figures.epsilon_check(summary), key="fig_eps_check")
    if private:
        names = dict(zip(summary["mechanism"], summary["mechanism_name"]))
        mechanism = st.selectbox(
            "ε spent during training of",
            private,
            format_func=lambda m: display_name(m, names.get(m)),
            key="acc_mechanism",
        )
        history = run.history
        if not history.empty:
            history = history[history["epsilon_target"].isin(summary["epsilon_target"])]
        common.plot(figures.epsilon_over_training(history, mechanism), key="fig_eps_training")
    st.markdown("**Calibration table**")
    common.table(calibration_table(summary))


def _fairness(run: Run, summary: pd.DataFrame) -> None:
    options = [c for c in FAIRNESS_COLUMNS if f"{c}_mean" in summary and summary[f"{c}_mean"].notna().any()]
    if not run.dataset.get("sensitive_attribute"):
        st.info("This dataset has no sensitive attribute, so only the gap between classes is measured.")
    if not options:
        st.info("No fairness metric was chosen for this run.")
        return
    column = st.segmented_control(
        "Fairness metric", options, default=options[0], format_func=column_label, key="fair_metric"
    )
    column = column or options[0]
    common.plot(figures.privacy_utility(summary, column), key="fig_fairness")
    groups = run.dataset.get("group_names")
    if groups and run.dataset.get("sensitive_attribute"):
        st.markdown(f"Sensitive attribute: **{run.dataset['sensitive_attribute']}** ({', '.join(groups)}).")


def _tables(run: Run, summary: pd.DataFrame) -> None:
    columns = [
        "test_accuracy",
        "test_f1_macro",
        "test_roc_auc",
        "test_loss",
        "epsilon_spent",
        "epsilon_worst_case",
        "noise_multiplier",
        "generalization_gap",
        *FAIRNESS_COLUMNS,
        "time_per_epoch_s",
    ]
    st.markdown("**Summary** (mean ± standard deviation over the seeds)")
    common.table(summary_table(summary, [c for c in columns if f"{c}_mean" in summary]))
    st.markdown("**All trainings**")
    results = (
        run.results[run.results["mechanism"].isin(summary["mechanism"])] if "mechanism" in run.results else run.results
    )
    common.table(results)
    left, middle, right = st.columns(3)
    for column, name, mime in ((left, RESULTS, "text/csv"), (middle, SUMMARY, "text/csv")):
        path = run.path / name
        if path.is_file():
            column.download_button(
                f"Download {name}",
                path.read_bytes(),
                file_name=f"{run.path.name}_{name}",
                mime=mime,
            )


def render() -> None:
    st.title("Results")
    run = common.run_selector()
    if run is None:
        return
    common.run_header(run)
    common.progress_banner(run)
    summary_all = run.summary()
    if summary_all.empty:
        st.warning("No successful training in this run yet.")
        common.show_failures(run)
        return
    summary = common.sidebar_filters(run, summary_all)
    if summary.empty:
        st.info("Nothing to show with the current filters.")
        return
    _kpis(summary_all)
    findings = key_findings(summary_all, run.results)
    if findings:
        with st.container(border=True):
            st.markdown("**Key findings**\n" + "\n".join(f"- {f}" for f in findings))
    common.show_failures(run)
    tabs = st.tabs(["Utility", "Privacy (ε spent)", "Fairness", "Tables"])
    with tabs[0]:
        _utility(run, summary)
    with tabs[1]:
        _accounting(run, summary)
    with tabs[2]:
        _fairness(run, summary)
    with tabs[3]:
        _tables(run, summary)
