"""Helpers shared by the dashboard pages: run discovery and loading, filters, headers, charts."""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dpbench.experiments.storage import HISTORY, RESULTS, STATUS, Run, find_runs, load_run, read_json
from dpbench.reporting.style import display_name, eps_text

PLOTLY_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}

#: Pages registered by app.py, so that views can link to each other.
PAGES: dict[str, Any] = {}

#: When ``dpbench dashboard`` was started (set by the command), else when this module was first imported:
#: only the runs launched since then are shown, so that the dashboard starts blank.
STARTED_AT = float(os.environ.get("DPBENCH_STARTED_AT") or time.time())


def run_roots() -> list[Path]:
    """Directories where runs are written and searched: ``$DPBENCH_RUNS_DIRS``, or ./runs."""
    value = os.environ.get("DPBENCH_RUNS_DIRS")
    roots = [Path(p) for p in value.split(os.pathsep) if p] if value else [Path("runs")]
    return [r.resolve() for r in roots]


def _signature(path: Path) -> tuple[float, ...]:
    return tuple(
        (path / name).stat().st_mtime if (path / name).exists() else 0.0 for name in (RESULTS, STATUS, HISTORY)
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _load_run(path: str, signature: tuple[float, ...]) -> Run:
    return load_run(path)


def get_run(path: str | Path) -> Run:
    return _load_run(str(path), _signature(Path(path)))


@st.cache_data(show_spinner=False, ttl=5)
def available_runs(roots: tuple[str, ...], kind: str = "ml") -> list[str]:
    """The runs launched since the dashboard started (older runs stay on disk but are not listed)."""
    recent = []
    for path in find_runs(list(roots), kind):
        started = read_json(path / STATUS).get("started")
        try:
            if datetime.fromisoformat(started).timestamp() >= STARTED_AT - 1:
                recent.append(str(path))
        except (TypeError, ValueError):
            continue
    return recent


def plot(fig: go.Figure, key: str | None = None) -> None:
    """``st.plotly_chart`` keeping DP-Bench's styling, with a fallback for older Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG, key=key)


def table(frame: pd.DataFrame | Any, **kwargs: Any) -> None:
    kwargs.setdefault("hide_index", True)
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(frame, use_container_width=True, **kwargs)


def run_selector() -> Run | None:
    """Sidebar selector of the run to analyse (shared by the Results and Mechanism pages)."""
    roots = tuple(str(r) for r in run_roots())
    runs = available_runs(roots)
    if not runs:
        st.info("No experiment yet: build one on the **Run an experiment** page, then come back here.")
        return None
    labels = {}
    for path in runs:
        try:
            labels[path] = get_run(path).label()
        except Exception:  # a run being written can be momentarily unreadable
            labels[path] = Path(path).name
    current = st.session_state.get("run_path")
    index = runs.index(current) if current in runs else 0
    choice = st.sidebar.selectbox("Run", runs, index=index, format_func=lambda p: labels.get(p, p), key="run_select")
    st.session_state["run_path"] = choice
    return get_run(choice)


def select_run(path: str) -> None:
    """Make ``path`` the run shown by the Results and Mechanism pages."""
    st.session_state["run_path"] = path
    st.session_state.pop("run_select", None)


def run_header(run: Run) -> None:
    st.subheader(run.name)
    dataset = run.dataset
    model = run.config.get("model", {})
    training = run.config.get("training", {})
    ok = int((run.results["status"] == "ok").sum()) if "status" in run.results else 0
    failed = len(run.results) - ok
    facts = [
        f"**Dataset** {dataset.get('name', '?')} ({dataset.get('n_train', '?')} / {dataset.get('n_test', '?')})",
        f"**Model** {model.get('name', '?')}",
        f"**δ** {dataset.get('delta', float('nan')):.2g}",
        f"**Seeds** {len(run.config.get('seeds', []))}",
        f"**Epochs** {training.get('epochs', '?')} × batch {training.get('batch_size', '?')}",
        f"**Trainings** {ok} ok" + (f", {failed} failed" if failed else ""),
    ]
    st.markdown(" · ".join(facts))


def progress_banner(run: Run) -> None:
    status = run.status
    if status.get("state") != "running":
        if status.get("state") in ("failed", "interrupted"):
            st.error(f"This run {status['state']}: {status.get('error', '')}")
        return
    total, done = int(status.get("total") or 0), int(status.get("done") or 0)
    st.progress(
        done / total if total else 0.0, text=f"Running: {done}/{total} trainings done, now {status.get('current')}"
    )
    if st.button("Refresh"):
        st.rerun()


def sidebar_filters(run: Run, summary: pd.DataFrame) -> pd.DataFrame:
    """Mechanism and budget filters in the sidebar; returns the filtered summary."""
    names = dict(zip(summary["mechanism"], summary["mechanism_name"]))
    mechanisms = [m for m in run.mechanisms if m in names]
    chosen = st.sidebar.multiselect(
        "Mechanisms",
        mechanisms,
        default=mechanisms,
        format_func=lambda m: display_name(m, names.get(m)),
        key=f"filter_mechanisms_{run.path.name}",
    )
    epsilons = sorted(e for e in summary["epsilon_target"].unique() if math.isfinite(e))
    chosen_eps = st.sidebar.multiselect(
        "Privacy budgets ε", epsilons, default=epsilons, format_func=eps_text, key=f"filter_eps_{run.path.name}"
    )
    keep = summary["mechanism"].isin(chosen) & (
        summary["epsilon_target"].isin(chosen_eps) | ~summary["epsilon_target"].apply(math.isfinite)
    )
    return summary[keep]


def show_failures(run: Run) -> None:
    if "status" not in run.results:
        return
    failed = run.results[run.results["status"] != "ok"]
    if not failed.empty:
        with st.expander(f"{len(failed)} failed training(s)"):
            table(failed[["run_id", "error"]])


def read_status(run_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(run_dir) / STATUS)
