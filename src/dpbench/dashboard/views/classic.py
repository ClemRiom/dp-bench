"""Classic DP page: statistical queries on databases, answered by every compatible mechanism.

Two ways to start a benchmark:

* **One database** -- choose a table, edit the list of queries, choose the mechanisms;
* **Compare databases** -- the same types of queries (count, mean, median, histogram,
  range queries...) on several tables, to see how the size of a database changes the error.

The results are shown below the tabs (they stay until the next benchmark).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from dpbench.classic.runner import (
    AUTO,
    ClassicConfig,
    ClassicResult,
    Spec,
    compatible,
    generic_queries,
    run_query_benchmark,
)
from dpbench.classic.tables import Table
from dpbench.core.registry import QUERIES, QUERY_MECHANISMS, TABLES
from dpbench.core.schema import CATEGORICAL, NUMERIC
from dpbench.dashboard import common
from dpbench.reporting import classic_figures

EPSILONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
METRICS = {"rel_error": "Relative error", "abs_error": "Absolute error"}
COMPARISON_TABLES = ("census_synthetic", "diabetes")


def _title(registry, name: str) -> str:
    return getattr(registry.get(name), "title", "") or name


@st.cache_data(show_spinner="Loading the table...", max_entries=8)
def _load_table(name: str) -> Table:
    return TABLES.get(name)().load()


def default_queries(table: Table) -> list[dict[str, Any]]:
    """The generic queries of the table (``generic_queries``), labelled with their description."""
    return [
        {"name": s.name, "params": dict(s.params), "label": QUERIES.get(s.name)(**s.params).describe()}
        for s in generic_queries(table)
    ]


def _table_section() -> Table | None:
    names = TABLES.names()
    if not names:
        st.error("No table is registered.")
        return None
    default = names.index("census_synthetic") if "census_synthetic" in names else 0
    name = st.selectbox(
        "Database (table)",
        names,
        index=default,
        format_func=lambda n: f"{n} — {(TABLES.get(n).__doc__ or '').strip().splitlines()[0][:90]}",
    )
    try:
        table = _load_table(name)
    except Exception as error:
        st.error(f"Cannot load the table {name!r}: {error}")
        return None
    st.caption(
        f"**{table.n:,} records** · {len(table.numeric_columns())} numeric and {len(table.categorical_columns())} "
        f"categorical columns · {table.description}"
    )
    if table.schema.bounds_from_data:
        st.warning(
            "The domains of "
            + ", ".join(f"`{c}`" for c in table.schema.bounds_from_data)
            + " were computed from the data. Differential privacy assumes public domains: in a real deployment "
            "they must come from documentation or domain knowledge.",
            icon=":material/warning:",
        )
    with st.expander("Columns and their public domains"):
        rows = []
        for column in table.schema.columns.values():
            if column.kind == NUMERIC:
                domain = f"[{column.lower:g}, {column.upper:g}]"
            elif column.kind == CATEGORICAL:
                cats = column.categories or []
                domain = ", ".join(cats[:8]) + (f", … ({len(cats)} values)" if len(cats) > 8 else "")
            else:
                domain = "ignored"
            rows.append({"Column": column.name, "Type": column.kind, "Public domain": domain})
        common.table(pd.DataFrame(rows))
        common.table(table.frame.head(8))
    return table


def _query_form(table: Table) -> dict[str, Any] | None:
    """Form to add one query; returns its spec when the button is pressed."""
    names = QUERIES.names()
    c1, c2 = st.columns([1, 2])
    kind = c1.selectbox("Query type", names, format_func=lambda n: _title(QUERIES, n), key="new_query_type")
    c2.caption((QUERIES.get(kind).__doc__ or "").strip().split("\n\n")[0])
    numeric, categorical = table.numeric_columns(), table.categorical_columns()
    accepted = QUERIES.get(kind).column_kinds
    columns = [c for c in table.schema.columns if table.column(c).kind in accepted]
    params: dict[str, Any] = {}
    c1, c2, c3, c4 = st.columns(4)
    if kind == "count":
        column = c1.selectbox("Condition on", ["(all records)", *numeric, *categorical], key="nq_count_col")
        if column != "(all records)":
            params["column"] = column
            params["operator"] = c2.selectbox("Operator", ["==", "!=", "<", "<=", ">", ">="], index=4, key="nq_op")
            spec = table.column(column)
            if spec.kind == NUMERIC:
                params["value"] = c3.number_input(
                    "Value", value=float(round((spec.lower + spec.upper) / 2, 6)), key="nq_val"
                )
            else:
                params["value"] = c3.selectbox("Value", spec.categories or [], key="nq_cat")
    elif kind in ("marginal", "centroid"):
        choices = numeric if kind == "centroid" else [*categorical, *numeric]
        picked = c1.multiselect(
            "Columns",
            choices,
            default=choices[:2],
            max_selections=2 if kind == "marginal" else None,
            key=f"nq_cols_{kind}",
        )
        params["columns"] = picked
        if kind == "marginal":
            params["bins"] = int(c2.number_input("Bins (numeric columns)", 2, 50, 5, key="nq_mbins"))
    else:
        if not columns:
            st.info("No column of a suitable type in this table.")
            return None
        column = c1.selectbox("Column", columns, key=f"nq_col_{kind}")
        params["column"] = column
        is_numeric = table.column(column).kind == NUMERIC
        if kind == "quantile":
            params["q"] = c2.slider("Quantile q", 0.05, 0.95, 0.5, 0.05, key="nq_q")
        if kind in ("histogram", "mode") and is_numeric:
            params["bins"] = int(c2.number_input("Bins", 2, 500, 10, key=f"nq_bins_{kind}"))
        if kind == "range_queries":
            if is_numeric:
                params["bins"] = int(c2.number_input("Cells (domain size)", 4, 4096, 64, key="nq_rbins"))
            params["kind"] = c3.selectbox("Workload", ["random", "prefix", "all"], key="nq_rkind")
            if params["kind"] == "random":
                params["n_queries"] = int(c4.number_input("Number of ranges", 1, 5000, 100, key="nq_rn"))
    label = st.text_input("Label (optional)", "", key="nq_label", placeholder="e.g. Median income")
    if st.button("Add this query", icon=":material/add:"):
        try:
            query = QUERIES.get(kind)(**params)
            query.tasks(table)  # validates the columns and the parameters
        except Exception as error:
            st.error(f"Invalid query: {error}")
            return None
        return {"name": kind, "params": params, "label": label.strip() or query.describe()}
    return None


def _queries_section(table: Table) -> list[dict[str, Any]]:
    store = st.session_state.setdefault("classic_queries", {})
    if table.name not in store:
        store[table.name] = default_queries(table)
    queries = store[table.name]
    for i, spec in enumerate(list(queries)):
        left, middle, right = st.columns([4, 3, 1])
        left.markdown(f"**{spec['label']}**")
        middle.caption(f"{_title(QUERIES, spec['name'])} · " + ", ".join(f"{k}={v}" for k, v in spec["params"].items()))
        if right.button("Remove", key=f"rm_query_{table.name}_{i}", icon=":material/close:"):
            queries.pop(i)
            st.rerun()
    with st.expander("Add a query", icon=":material/add_circle:"):
        added = _query_form(table)
        if added is not None:
            labels = {q["label"] for q in queries}
            if added["label"] in labels:
                added["label"] += f" ({len(queries) + 1})"
            queries.append(added)
            st.rerun()
    if st.button("Reset to the default queries", icon=":material/restart_alt:"):
        store[table.name] = default_queries(table)
        st.rerun()
    return queries


def _compatibility(queries: list[dict[str, Any]], mechanisms: list[str]) -> pd.DataFrame:
    rows = []
    for spec in queries:
        kind = QUERIES.get(spec["name"]).task_kind
        row = {"Query": spec["label"], "Kind": kind}
        for name in mechanisms:
            row[_title(QUERY_MECHANISMS, name)] = "✓" if kind in QUERY_MECHANISMS.get(name).supports else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _budget_inputs(key: str, default_repetitions: int = 30) -> tuple[list[float], float, int, int]:
    c1, c2, c3, c4 = st.columns(4)
    epsilons = c1.multiselect("Budgets ε", EPSILONS, default=[0.1, 0.5, 1.0, 2.0, 4.0], key=f"eps_{key}")
    delta = c2.number_input(
        "δ (Gaussian mechanisms)", min_value=1e-12, max_value=0.1, value=1e-6, format="%.1e", key=f"delta_{key}"
    )
    repetitions = c3.slider(
        "Repetitions",
        5,
        200,
        default_repetitions,
        help="Each configuration is released this many times.",
        key=f"reps_{key}",
    )
    seed = int(c4.number_input("Seed", 0, 10_000, 0, key=f"seed_{key}"))
    return sorted(epsilons), float(delta), int(repetitions), seed


def _launch(config: ClassicConfig, tables: list[Table]) -> None:
    bar = st.progress(0.0, text="Starting...")
    result = run_query_benchmark(config, tables, progress=lambda f, text: bar.progress(min(f, 1.0), text=text))
    bar.empty()
    st.session_state["classic_result"] = {"config": config, "result": result}


def _one_database() -> None:
    table = _table_section()
    if table is None:
        return
    st.markdown("#### Queries")
    queries = _queries_section(table)
    st.markdown("#### Mechanisms and budget")
    names = QUERY_MECHANISMS.names()
    mechanisms = st.multiselect(
        "Mechanisms", names, default=names, format_func=lambda n: f"{_title(QUERY_MECHANISMS, n)} ({n})"
    )
    if queries and mechanisms:
        with st.expander("Which mechanism can answer which query"):
            common.table(_compatibility(queries, mechanisms))
    epsilons, delta, repetitions, seed = _budget_inputs("one")
    usable = [q for q in queries if any(compatible(QUERIES.get(q["name"])(**q["params"]), m) for m in mechanisms)]
    if st.button("Run the benchmark", type="primary", icon=":material/play_arrow:", disabled=not (usable and epsilons)):
        config = ClassicConfig(
            name=f"classic-{table.name}",
            description=f"{len(queries)} queries × {len(mechanisms)} mechanisms on {table.name}",
            tables=[Spec(table.name)],
            queries=[Spec(q["name"], dict(q["params"]), q["label"]) for q in queries],
            mechanisms=[Spec(m, {}, _title(QUERY_MECHANISMS, m)) for m in mechanisms],
            epsilons=epsilons,
            delta=delta,
            repetitions=repetitions,
            seed=seed,
        )
        _launch(config, [table])


def _compare_databases() -> None:
    st.caption(
        "Every table gets the same types of queries — count, count with a condition, sum, mean, median, 100 range "
        "queries, centroid, histogram, mode, 2-way marginal — on columns of the right type (chosen from the schema, "
        "never from the data). The errors can then be compared from one database to the other."
    )
    names = TABLES.names()
    tables = st.multiselect(
        "Databases",
        names,
        default=[n for n in COMPARISON_TABLES if n in names],
    )
    defaults = ["laplace", "gaussian", "exponential", "polar_laplace"]
    mechanisms = st.multiselect(
        "Mechanisms",
        QUERY_MECHANISMS.names(),
        default=[m for m in defaults if m in QUERY_MECHANISMS],
        format_func=lambda n: f"{_title(QUERY_MECHANISMS, n)} ({n})",
        key="compare_mechanisms",
    )
    epsilons, delta, repetitions, seed = _budget_inputs("compare", default_repetitions=20)
    ready = bool(tables and mechanisms and epsilons)
    if st.button("Compare the databases", type="primary", icon=":material/compare_arrows:", disabled=not ready):
        loaded = []
        for name in tables:
            try:
                loaded.append(_load_table(name))
            except Exception as error:
                st.error(f"Cannot load the table {name!r}: {error}")
                return
        config = ClassicConfig(
            name="classic-comparison",
            description=f"{len(tables)} databases × {len(mechanisms)} mechanisms (generic queries)",
            tables=[Spec(t) for t in tables],
            queries=AUTO,
            mechanisms=[Spec(m, {}, _title(QUERY_MECHANISMS, m)) for m in mechanisms],
            epsilons=epsilons,
            delta=delta,
            repetitions=repetitions,
            seed=seed,
        )
        _launch(config, loaded)


def _comparison(summary: pd.DataFrame) -> None:
    """Errors of every mechanism against the size of the database, for one type of query."""
    st.markdown("#### The databases compared")
    left, middle, right = st.columns([3, 3, 2])
    queries = list(dict.fromkeys(summary["query"]))
    query = left.selectbox("Query", queries, key="compare_query")
    epsilons = sorted(summary["epsilon"].unique())
    epsilon = middle.select_slider(
        "Budget ε", epsilons, value=1.0 if 1.0 in epsilons else epsilons[len(epsilons) // 2], key="compare_eps"
    )
    metric = right.segmented_control(
        "Error", list(METRICS), default="rel_error", format_func=METRICS.get, key="compare_metric"
    )
    metric = metric or "rel_error"
    common.plot(classic_figures.error_vs_records(summary, query, epsilon, metric), key="compare_records")
    st.caption(
        "One point per database, placed at its number of records. For counts, sums and histograms the noise does "
        "not depend on the size of the database, so the relative error falls like 1/n: the same ε costs much more "
        "accuracy on a small database."
    )
    rows = summary[(summary["query"] == query) & (summary["epsilon"] == epsilon)]
    view = rows.pivot_table(index="mechanism", columns="table", values=f"{metric}_mean", sort=False)
    view = view[[t for t in dict.fromkeys(summary["table"]) if t in view.columns]]
    common.table(view.map(classic_figures.compact), hide_index=False)
    columns = rows.drop_duplicates("table")[["table", "records", "query_description"]]
    st.caption("Columns used: " + "; ".join(f"{r.table}: {r.query_description}" for r in columns.itertuples()))


def _details(config: ClassicConfig, result: ClassicResult, summary: pd.DataFrame, results: pd.DataFrame) -> None:
    """Heatmap of every query x mechanism, then one query in detail (one table)."""
    ok = results[results["status"] == "ok"]
    st.markdown("#### Every query against every mechanism")
    left, right = st.columns([3, 1])
    epsilons = sorted(summary["epsilon"].unique())
    epsilon = left.select_slider(
        "Budget ε for the overview",
        epsilons,
        value=1.0 if 1.0 in epsilons else epsilons[len(epsilons) // 2],
        key="classic_eps",
    )
    metric = right.segmented_control(
        "Error", list(METRICS), default="rel_error", format_func=METRICS.get, key="classic_metric"
    )
    metric = metric or "rel_error"
    common.plot(classic_figures.error_heatmap(summary, epsilon, metric), key="classic_heatmap")
    st.caption(
        "Each cell is the mean error of a mechanism on a query over the repetitions. Relative error: |error| / "
        "|exact answer| for numbers; L1 error / total for histograms; mean error per range / number of records for "
        "range queries; share of the records between the answer and the true value for quantiles; distance / "
        "domain diagonal for centroids; regret / count of the mode for the mode. The colour compares the "
        "mechanisms of a row (the best one is the lightest). A blank cell: the mechanism cannot answer this kind "
        "of query."
    )
    st.markdown("#### One query in detail")
    queries = list(dict.fromkeys(summary["query"]))
    query = st.selectbox("Query", queries, key="classic_query")
    rows = summary[summary["query"] == query]
    st.markdown(f"**{METRICS[metric]} against ε** (mean over the repetitions) · {rows['query_description'].iloc[0]}")
    common.plot(classic_figures.error_vs_epsilon(summary, query, metric), key="classic_curve")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Exact answer:** `{rows['exact'].iloc[0]}`")
        sample = ok[(ok["query"] == query) & (ok["repetition"] == 0)][["mechanism", "epsilon", "estimate"]]
        with st.expander("A released answer of every mechanism (first repetition)"):
            common.table(
                sample.rename(columns={"mechanism": "Mechanism", "epsilon": "ε", "estimate": "Released answer"})
            )
    with right:
        if "correct_mean" in rows and rows["correct_mean"].notna().any():
            st.markdown("**Fraction of correct answers against ε**")
            common.plot(classic_figures.error_vs_epsilon(summary, query, "correct"), key="classic_correct")
    columns = {
        "mechanism": "Mechanism",
        "epsilon": "ε",
        "rel_error_mean": "Relative error (mean)",
        "rel_error_std": "(std)",
        "abs_error_mean": "Absolute error (mean)",
        "abs_error_median": "(median)",
        "epsilon_spent": "ε spent",
        "delta_spent": "δ spent",
        "guarantee": "Guarantee",
        "time_ms_mean": "ms per release",
    }
    table = rows[[c for c in columns if c in rows]].rename(columns=columns)
    common.table(table.round(6))


def _results(config: ClassicConfig, result: ClassicResult) -> None:
    summary, results = result.summary, result.results
    failed = results[results["status"] != "ok"] if not results.empty else pd.DataFrame()
    if summary.empty:
        st.warning("No release succeeded.")
        if not failed.empty:
            common.table(failed[["query", "mechanism", "epsilon", "status", "error"]].drop_duplicates())
        return
    ok = results[results["status"] == "ok"]
    tables = list(dict.fromkeys(summary["table"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Releases", f"{len(ok):,}", border=True)
    c2.metric("Databases × queries", f"{len(tables)} × {summary['query'].nunique()}", border=True)
    c3.metric("Mechanisms", summary["mechanism"].nunique(), border=True)
    gap = (summary["epsilon_spent"] - summary["epsilon"]).abs().max()
    c4.metric(
        "ε spent = target ε",
        "yes" if gap < 1e-9 else f"max gap {gap:.2g}",
        border=True,
        help="Each query splits its budget between its parts (basic composition): the ε spent must equal the target.",
    )
    if len(tables) > 1:
        _comparison(summary)
        st.markdown("#### One database in detail")
        table = st.selectbox(
            "Database",
            tables,
            format_func=lambda t: f"{t} ({int(summary.loc[summary['table'] == t, 'records'].iloc[0]):,} records)",
            key="classic_detail_table",
        )
    else:
        table = tables[0]
    _details(config, result, summary[summary["table"] == table], results[results["table"] == table])
    if result.skipped:
        with st.expander(f"{len(result.skipped)} incompatible (query, mechanism) pairs, not run"):
            st.markdown("\n".join(f"- {s}" for s in result.skipped))
    if not failed.empty:
        keys = [c for c in ("table", "query", "mechanism", "epsilon") if c in failed]
        with st.expander(
            f"{failed[keys].drop_duplicates().shape[0]} skipped or failed configurations", icon=":material/error:"
        ):
            common.table(failed[[*keys, "status", "error"]].drop_duplicates(keys))
    left, middle = st.columns(2)
    left.download_button(
        "Download the summary (CSV)",
        summary.to_csv(index=False),
        file_name="classic_summary.csv",
        icon=":material/download:",
    )
    middle.download_button(
        "Download every release (CSV)",
        results.to_csv(index=False),
        file_name="classic_results.csv",
        icon=":material/download:",
    )


def render() -> None:
    st.title("Classic DP: queries on databases")
    st.caption(
        "Statistical queries (counts, sums, means, histograms, quantiles, range queries...) answered with the "
        "classic DP mechanisms. Every compatible (query, mechanism) pair is released many times for each ε; the "
        "error is averaged over the repetitions. Neighbouring databases differ by one record (added or removed)."
    )
    one, many = st.tabs(["One database", "Compare databases"])
    with one:
        _one_database()
    with many:
        _compare_databases()
    state = st.session_state.get("classic_result")
    if state is not None:
        st.divider()
        st.subheader("Results")
        config: ClassicConfig = state["config"]
        names = ", ".join(t.name for t in config.tables)
        st.caption(f"{names} · ε ∈ {config.epsilons} · δ = {config.delta:g} · {config.repetitions} repetitions")
        _results(config, state["result"])
