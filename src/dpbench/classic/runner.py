"""Benchmark of classic DP on databases: every table x query x compatible mechanism x epsilon, repeated.

The mechanisms are random, so each configuration is released ``repetitions`` times with
independent seeds and the error is averaged. The exact answers are computed once per
query. The dashboard builds the configuration (:class:`ClassicConfig`) from its forms and
keeps the result in memory.

Several tables can be compared in one run. With ``queries: auto``, every table gets the
same *types* of queries (count, mean, median, histogram, range queries...) on columns of
the right type (:func:`generic_queries`), so that the databases can be compared.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dpbench.classic.queries import Query
from dpbench.classic.tables import Table
from dpbench.core.registry import QUERIES, QUERY_MECHANISMS, TABLES
from dpbench.core.seeding import derive_seed

#: ``queries: auto``: the generic set of queries of :func:`generic_queries`, built for every table.
AUTO = "auto"


@dataclass
class Spec:
    """A plugin name, its parameters and a display label."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


@dataclass
class ClassicConfig:
    name: str
    tables: list[Spec]
    #: The queries asked to every table, or ``"auto"`` (the generic set of each table).
    queries: list[Spec] | str
    mechanisms: list[Spec]
    epsilons: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0])
    delta: float = 1e-6
    repetitions: int = 30
    seed: int = 0
    description: str = ""

    @property
    def table(self) -> Spec:
        """The first table (the only one of most benchmarks)."""
        return self.tables[0]


def generic_queries(table: Table) -> list[Spec]:
    """A varied set of queries chosen from the column types only (never from the data).

    The labels are the query types ("Mean", "Median"...), the same for every table, so that
    several databases can be compared; the column used is in the query's description. The
    median is asked on the numeric column with the widest public domain (usually the most
    continuous one: on small integer columns such as ages, a median is rarely wrong).
    """
    numeric, categorical = table.numeric_columns(), table.categorical_columns()
    queries = [Spec("count", {}, "Count")]
    if numeric:
        first = table.column(numeric[0])
        middle = round((first.lower + first.upper) / 2, 6)
        widest = max(numeric, key=lambda c: table.column(c).width)
        queries += [
            Spec("count", {"column": first.name, "operator": ">=", "value": middle}, "Count with a condition"),
            Spec("sum", {"column": first.name}, "Sum"),
            Spec("mean", {"column": first.name}, "Mean"),
            Spec("quantile", {"column": widest, "q": 0.5}, "Median"),
            Spec("range_queries", {"column": first.name, "bins": 64, "n_queries": 100}, "100 range queries"),
        ]
    if len(numeric) > 1:
        queries.append(Spec("centroid", {"columns": numeric[:2]}, "Centroid of 2 columns"))
    if categorical:
        richest = max(categorical, key=lambda c: len(table.column(c).categories or []))
        queries += [Spec("histogram", {"column": richest}, "Histogram"), Spec("mode", {"column": richest}, "Mode")]
    if len(categorical) > 1:
        queries.append(Spec("marginal", {"columns": categorical[:2]}, "2-way marginal"))
    return queries


def table_queries(config: ClassicConfig, table: Table) -> list[Spec]:
    """The queries asked to a table: those of the configuration, or its generic set."""
    return generic_queries(table) if config.queries == AUTO else list(config.queries)


def compatible(query: Query, mechanism_name: str) -> bool:
    """Whether a mechanism supports the kind of tasks a query produces."""
    return query.task_kind in QUERY_MECHANISMS.get(mechanism_name).supports


@dataclass
class ClassicResult:
    results: pd.DataFrame
    summary: pd.DataFrame
    skipped: list[str]
    #: ``Table.summary()`` of every table.
    tables: list[dict[str, Any]]


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    ok = results[results["status"] == "ok"]
    if ok.empty:
        return pd.DataFrame()
    keys = ["table", "query", "query_name", "mechanism", "mechanism_name", "epsilon"]
    metrics = [c for c in ("abs_error", "rel_error", "max_error", "correct", "time_ms") if c in ok]
    grouped = ok.groupby(keys, sort=False)
    summary = grouped[metrics].agg(["mean", "std", "median"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    first = grouped[["records", "query_description", "epsilon_spent", "delta_spent", "exact", "guarantee"]].first()
    summary = summary.join(first).reset_index()
    summary["repetitions"] = grouped.size().to_numpy()
    return summary


def load_tables(config: ClassicConfig) -> list[Table]:
    return [TABLES.get(t.name)(**t.params).load() for t in config.tables]


def run_query_benchmark(
    config: ClassicConfig,
    tables: Table | list[Table] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> ClassicResult:
    """Release every (table, query, compatible mechanism, epsilon) ``repetitions`` times and measure the errors.

    ``tables`` (already loaded) replace the tables of the configuration.
    """
    if tables is None:
        tables = load_tables(config)
    elif isinstance(tables, Table):
        tables = [tables]
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    plans = []
    for current in tables:
        for qspec in table_queries(config, current):
            query = QUERIES.get(qspec.name)(**qspec.params)
            try:
                tasks = query.tasks(current)  # fails if the table has no such column
                exact = query.exact(current, tasks)
            except (KeyError, ValueError) as error:
                where = f"{qspec.label} on table {current.name}"
                skipped.append(f"{where}: the query does not apply ({type(error).__name__}: {error})")
                continue
            for mspec in config.mechanisms:
                if not compatible(query, mspec.name):
                    if len(tables) == 1 or current is tables[0]:
                        skipped.append(f"{mspec.label} cannot answer {qspec.label} ({query.task_kind} query)")
                    continue
                plans.append((current, qspec, query, tasks, exact, mspec))
    total = max(len(plans) * len(config.epsilons), 1)
    done = 0
    for current, qspec, query, tasks, exact, mspec in plans:
        mechanism = QUERY_MECHANISMS.get(mspec.name)(**mspec.params)
        exact_text = _short(exact)
        for epsilon in config.epsilons:
            done += 1
            if progress is not None:
                prefix = f"{current.name} · " if len(tables) > 1 else ""
                progress(done / total, f"{prefix}{qspec.label} · {mspec.label} · ε = {epsilon:g}")
            # Basic composition: the query's tasks share the budget equally.
            eps_task, delta_task = epsilon / len(tasks), config.delta / len(tasks)
            base = {
                "table": current.name,
                "records": current.n,
                "query": qspec.label,
                "query_name": qspec.name,
                "query_description": query.describe(),
                "mechanism": mspec.label,
                "mechanism_name": mspec.name,
                "epsilon": epsilon,
                "delta": config.delta,
                "exact": exact_text,
            }
            try:
                for task in tasks:
                    mechanism.check(task, eps_task, delta_task)
            except ValueError as error:
                rows.append({**base, "repetition": 0, "status": "skipped", "error": str(error)})
                continue
            spent = [mechanism.guarantee(eps_task, delta_task) for _ in tasks]
            guarantee = "(ε, δ)-DP" if mechanism.uses_delta else "pure ε-DP"
            for repetition in range(config.repetitions):
                seed = derive_seed(config.seed, current.name, qspec.label, mspec.label, epsilon, repetition)
                rng = np.random.default_rng(seed)
                start = time.perf_counter()
                try:
                    outputs = [mechanism.release(task, eps_task, delta_task, rng) for task in tasks]
                    estimate = query.combine(outputs)
                    seconds = time.perf_counter() - start
                    errors = query.errors(estimate, exact, current, tasks)
                except Exception as error:  # a broken (user) mechanism must not stop the benchmark
                    message = f"{type(error).__name__}: {error}"
                    rows.append({**base, "repetition": repetition, "status": "failed", "error": message})
                    break
                rows.append(
                    {
                        **base,
                        "repetition": repetition,
                        "status": "ok",
                        "error": "",
                        "estimate": _short(estimate),
                        **errors,
                        "time_ms": 1000 * seconds,
                        "epsilon_spent": sum(e for e, _ in spent),
                        "delta_spent": sum(d for _, d in spent),
                        "guarantee": guarantee,
                    }
                )
    results = pd.DataFrame(rows)
    return ClassicResult(results, _summarize(results), skipped, [t.summary() for t in tables])


def _short(value: Any) -> str:
    """A compact text form of an answer (for the tables)."""
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if array.size == 1:
        return f"{array[0]:.6g}"
    shown = ", ".join(f"{v:.4g}" for v in array[:6])
    return f"[{shown}{', ...' if array.size > 6 else ''}] ({array.size} values)"
