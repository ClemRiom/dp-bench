"""DP-Bench dashboard (Streamlit). Start it with ``dpbench dashboard``.

The dashboard starts empty: everything is built from its pages (mechanisms, datasets,
budgets, queries), and only the experiments launched since it started are shown.
"""

from __future__ import annotations

import os

import streamlit as st

from dpbench import workspace
from dpbench.core.registry import load_plugins
from dpbench.dashboard import common
from dpbench.dashboard.views import accountants, add_algorithm, add_dataset, classic, results, run

st.set_page_config(page_title="DP-Bench", layout="wide")


@st.cache_resource(show_spinner=False)
def _startup() -> dict[str, str]:
    """Load the external plugins and the workspace once per dashboard process."""
    extra = os.environ.get("DPBENCH_PLUGINS")
    if extra:
        load_plugins([p for p in extra.split(os.pathsep) if p])
    return workspace.load_workspace()


errors = _startup()

common.PAGES.update(
    run=st.Page(run.render, title="Run an experiment", url_path="run", default=True),
    results=st.Page(results.render, title="Results", url_path="results"),
    accountants=st.Page(accountants.render, title="Accountants", url_path="accountants"),
    classic=st.Page(classic.render, title="Queries on databases", url_path="classic"),
    add_dataset=st.Page(add_dataset.render, title="Add a dataset", url_path="add-dataset"),
    add_algorithm=st.Page(add_algorithm.render, title="Add an algorithm", url_path="add-algorithm"),
)

navigation = st.navigation(
    {
        "Machine learning": [common.PAGES["run"], common.PAGES["results"], common.PAGES["accountants"]],
        "Classic DP": [common.PAGES["classic"]],
        "Add your own": [common.PAGES["add_dataset"], common.PAGES["add_algorithm"]],
    }
)
if errors:
    with st.sidebar.expander(f"{len(errors)} workspace file(s) could not be loaded"):
        for name, message in errors.items():
            st.markdown(f"**{name}**: {message}")
navigation.run()
