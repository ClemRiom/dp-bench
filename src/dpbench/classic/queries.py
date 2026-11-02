"""Query plugins: the statistics asked on a table.

A query turns a table into one or more tasks (``tasks.py``) and, after the mechanism has
released them, combines the noisy outputs into the final answer. Most queries are one task;
the mean is two (a noisy sum divided by a noisy count), and the privacy budget is split
equally between the tasks, so the whole query is (epsilon, delta)-DP by basic composition.

Every query also measures the error of an answer against the exact one:

* ``abs_error`` -- in the units of the answer (records, euros, years...);
* ``rel_error`` -- a scale-free version, defined for each query below, to compare queries.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from dpbench.classic.tables import Table
from dpbench.classic.tasks import (
    NUMERIC,
    QUANTILE,
    SELECTION,
    WORKLOAD,
    NumericTask,
    QuantileTask,
    SelectionTask,
    Task,
    WorkloadTask,
)
from dpbench.core.registry import register_query
from dpbench.core.schema import CATEGORICAL, ColumnSpec, category_codes, numeric_values
from dpbench.core.schema import NUMERIC as NUMERIC_COLUMN

OPERATORS = ("==", "!=", "<", "<=", ">", ">=")


def exact_output(task: Task) -> Any:
    """What a perfect (non-private) mechanism would release for a task."""
    if task.kind == NUMERIC:
        return task.answer.copy()
    if task.kind == SELECTION:
        return int(np.argmax(task.scores))
    if task.kind == QUANTILE:
        return float(np.quantile(task.values, task.q)) if len(task.values) else (task.lower + task.upper) / 2
    return task.answer


class Query(ABC):
    """Base class of the query plugins."""

    name: ClassVar[str]
    title: ClassVar[str] = ""
    #: Kind of the tasks the query produces (the mechanisms that can answer it support this kind).
    task_kind: ClassVar[str] = NUMERIC
    #: Column types accepted by the query's ``column`` parameter (for the dashboard).
    column_kinds: ClassVar[tuple[str, ...]] = (NUMERIC_COLUMN,)

    @abstractmethod
    def tasks(self, table: Table) -> list[Task]:
        """The tasks to release (the budget is split equally between them)."""

    def combine(self, outputs: list[Any]) -> Any:
        """The answer of the query from the released outputs of its tasks."""
        return outputs[0]

    def exact(self, table: Table, tasks: list[Task] | None = None) -> Any:
        """The exact (non-private) answer."""
        return self.combine([exact_output(t) for t in (tasks or self.tasks(table))])

    def errors(self, estimate: Any, exact: Any, table: Table, tasks: list[Task]) -> dict[str, float]:
        """Error metrics of a released answer; scalar answers by default."""
        estimate, exact = float(np.ravel(estimate)[0]), float(np.ravel(exact)[0])
        error = abs(estimate - exact)
        return {"abs_error": error, "rel_error": error / abs(exact) if exact else error}

    def describe(self) -> str:
        return self.title or self.name


def _bounds(column: ColumnSpec, lower: float | None, upper: float | None) -> tuple[float, float]:
    if column.kind != NUMERIC_COLUMN:
        raise ValueError(f"column {column.name!r} is not numeric")
    low = column.lower if lower is None else float(lower)
    high = column.upper if upper is None else float(upper)
    if not low < high:
        raise ValueError(f"column {column.name!r}: lower bound {low} must be below the upper bound {high}")
    return low, high


def _cell_codes(table: Table, column: str, bins: int) -> tuple[np.ndarray, list[str]]:
    """Cell of every record (-1 if missing) and the cell labels: the categories, or ``bins`` equal bins."""
    spec = table.column(column)
    if spec.kind == CATEGORICAL:
        return category_codes(table.frame, spec), list(spec.categories or [])
    if spec.kind != NUMERIC_COLUMN:
        raise ValueError(f"column {column!r} is ignored in the schema")
    import pandas as pd

    values = pd.to_numeric(table.frame[column], errors="coerce").to_numpy(dtype=np.float64)
    edges = np.linspace(spec.lower, spec.upper, int(bins) + 1)
    codes = np.clip(np.searchsorted(edges, np.clip(values, spec.lower, spec.upper), side="right") - 1, 0, bins - 1)
    labels = [f"[{edges[i]:.4g}, {edges[i + 1]:.4g}{']' if i == bins - 1 else ')'}" for i in range(int(bins))]
    return np.where(np.isnan(values), -1, codes), labels


def _cells(table: Table, column: str, bins: int) -> tuple[np.ndarray, list[str]]:
    """Cell index of the records with a value, and the cell labels."""
    codes, labels = _cell_codes(table, column, bins)
    return codes[codes >= 0], labels


def _counts(codes: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(codes, minlength=size).astype(np.float64)[:size]


class _ScalarQuery(Query):
    """A query answered by one number."""

    def combine(self, outputs: list[Any]) -> Any:
        return float(np.ravel(outputs[0])[0])


@register_query("count")
class Count(_ScalarQuery):
    """Number of records, or of the records whose column satisfies a condition (e.g. age >= 65). Sensitivity 1."""

    title = "Count"
    column_kinds = (NUMERIC_COLUMN, CATEGORICAL)

    def __init__(self, column: str | None = None, operator: str = "==", value: Any = None) -> None:
        if operator not in OPERATORS:
            raise ValueError(f"operator must be one of {OPERATORS}, got {operator!r}")
        self.column, self.operator, self.value = column, operator, value

    def mask(self, table: Table) -> np.ndarray:
        if self.column is None:
            return np.ones(table.n, dtype=bool)
        spec = table.column(self.column)
        if spec.kind == NUMERIC_COLUMN:
            import pandas as pd

            values = pd.to_numeric(table.frame[self.column], errors="coerce").to_numpy(dtype=np.float64)
            target = float(self.value)
        else:
            values = table.frame[self.column].astype(str).to_numpy()
            target = str(self.value)
        compare = {
            "==": np.equal,
            "!=": np.not_equal,
            "<": np.less,
            "<=": np.less_equal,
            ">": np.greater,
            ">=": np.greater_equal,
        }[self.operator]
        return compare(values, target)

    def tasks(self, table: Table) -> list[Task]:
        return [NumericTask([float(self.mask(table).sum())], 1.0, 1.0, integer=True)]

    def describe(self) -> str:
        if self.column is None:
            return "Count of the records"
        return f"Count of the records with {self.column} {self.operator} {self.value}"


@register_query("sum")
class Sum(_ScalarQuery):
    """Sum of a numeric column, each value clipped to [lower, upper]. Sensitivity max(|lower|, |upper|)."""

    title = "Sum"

    def __init__(self, column: str = "", lower: float | None = None, upper: float | None = None) -> None:
        self.column, self.lower, self.upper = column, lower, upper

    def tasks(self, table: Table) -> list[Task]:
        low, high = _bounds(table.column(self.column), self.lower, self.upper)
        values = np.clip(numeric_values(table.frame, table.column(self.column)), low, high)
        magnitude = max(abs(low), abs(high))
        return [NumericTask([values.sum()], magnitude, magnitude)]

    def describe(self) -> str:
        return f"Sum of {self.column}"


@register_query("mean")
class Mean(Query):
    """Mean of a numeric column: a noisy sum divided by a noisy count, each with half of the budget."""

    title = "Mean"

    def __init__(self, column: str = "", lower: float | None = None, upper: float | None = None) -> None:
        self.column, self.lower, self.upper = column, lower, upper
        self._range = (math.nan, math.nan)

    def tasks(self, table: Table) -> list[Task]:
        low, high = _bounds(table.column(self.column), self.lower, self.upper)
        self._range = (low, high)
        values = np.clip(numeric_values(table.frame, table.column(self.column)), low, high)
        magnitude = max(abs(low), abs(high))
        return [
            NumericTask([values.sum()], magnitude, magnitude),
            NumericTask([float(len(values))], 1.0, 1.0, integer=True),
        ]

    def combine(self, outputs: list[Any]) -> Any:
        total, count = float(np.ravel(outputs[0])[0]), float(np.ravel(outputs[1])[0])
        low, high = self._range
        return float(np.clip(total / max(count, 1.0), low, high))

    def describe(self) -> str:
        return f"Mean of {self.column}"


class _VectorQuery(Query):
    """Errors of vector answers: L1 error, relative to the total of the exact answer."""

    def errors(self, estimate: Any, exact: Any, table: Table, tasks: list[Task]) -> dict[str, float]:
        difference = np.asarray(estimate, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
        l1 = float(np.abs(difference).sum())
        return {
            "abs_error": l1,
            "rel_error": l1 / max(float(np.abs(exact).sum()), 1.0),
            "max_error": float(np.abs(difference).max()),
        }


@register_query("histogram")
class Histogram(_VectorQuery):
    """Counts per category (or per bin of a numeric column). L1 and L2 sensitivity 1."""

    title = "Histogram"
    column_kinds = (CATEGORICAL, NUMERIC_COLUMN)

    def __init__(self, column: str = "", bins: int = 10) -> None:
        self.column, self.bins = column, int(bins)

    def tasks(self, table: Table) -> list[Task]:
        codes, labels = _cells(table, self.column, self.bins)
        return [NumericTask(_counts(codes, len(labels)), 1.0, 1.0, integer=True, labels=labels)]

    def describe(self) -> str:
        return f"Histogram of {self.column}"


@register_query("marginal")
class Marginal(_VectorQuery):
    """Contingency table of two columns (counts of every pair of values). L1 and L2 sensitivity 1."""

    title = "2-way marginal"
    column_kinds = (CATEGORICAL, NUMERIC_COLUMN)

    def __init__(self, columns: list[str] | None = None, bins: int = 5) -> None:
        self.columns = list(columns or [])
        if len(self.columns) != 2:
            raise ValueError(f"a 2-way marginal needs exactly two columns, got {self.columns}")
        self.bins = int(bins)

    def tasks(self, table: Table) -> list[Task]:
        first, first_labels = _cell_codes(table, self.columns[0], self.bins)
        second, second_labels = _cell_codes(table, self.columns[1], self.bins)
        keep = (first >= 0) & (second >= 0)
        codes = first[keep] * len(second_labels) + second[keep]
        labels = [f"{a} × {b}" for a in first_labels for b in second_labels]
        return [NumericTask(_counts(codes, len(labels)), 1.0, 1.0, integer=True, labels=labels)]

    def describe(self) -> str:
        return f"Marginal of {self.columns[0]} × {self.columns[1]}"


@register_query("centroid")
class Centroid(Query):
    """Mean of several numeric columns (a point, e.g. a location): noisy vector sum / noisy count.

    The vector sum has L1 sensitivity sum_j max|bound_j| and L2 sensitivity
    sqrt(sum_j max|bound_j|^2): the query where the geometry of the noise (L1 or L2,
    coordinate-wise or radial + directional) matters. The error is the Euclidean distance
    to the exact centroid; ``rel_error`` divides it by the diagonal of the domain.
    """

    title = "Centroid (vector mean)"

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = list(columns or [])
        if not self.columns:
            raise ValueError("a centroid needs at least one numeric column")
        self._bounds: list[tuple[float, float]] = []

    def tasks(self, table: Table) -> list[Task]:
        specs = [table.column(c) for c in self.columns]
        if any(s.kind != NUMERIC_COLUMN for s in specs):
            raise ValueError(f"the centroid needs numeric columns, got {self.columns}")
        frame = table.frame[self.columns].apply(lambda s: s.astype(float)).dropna()
        self._bounds = [(s.lower, s.upper) for s in specs]
        values = np.column_stack(
            [np.clip(frame[c].to_numpy(), lo, hi) for c, (lo, hi) in zip(self.columns, self._bounds)]
        )
        magnitudes = np.array([max(abs(lo), abs(hi)) for lo, hi in self._bounds])
        return [
            NumericTask(
                values.sum(axis=0), float(magnitudes.sum()), float(np.sqrt((magnitudes**2).sum())), labels=self.columns
            ),
            NumericTask([float(len(values))], 1.0, 1.0, integer=True),
        ]

    def combine(self, outputs: list[Any]) -> Any:
        total, count = np.asarray(outputs[0], dtype=np.float64), float(np.ravel(outputs[1])[0])
        lows, highs = np.array([b[0] for b in self._bounds]), np.array([b[1] for b in self._bounds])
        return np.clip(total / max(count, 1.0), lows, highs)

    def errors(self, estimate: Any, exact: Any, table: Table, tasks: list[Task]) -> dict[str, float]:
        distance = float(np.linalg.norm(np.asarray(estimate) - np.asarray(exact)))
        diagonal = float(np.linalg.norm([hi - lo for lo, hi in self._bounds]))
        return {"abs_error": distance, "rel_error": distance / diagonal if diagonal else distance}

    def describe(self) -> str:
        return f"Centroid of ({', '.join(self.columns)})"


@register_query("range_queries")
class RangeQueries(Query):
    """Many range counting queries ("how many records have 30 <= age < 45?") on one column: a workload.

    The column is discretised into ``bins`` cells (or its categories, in order). The
    workload has ``n_queries`` random ranges (``kind="random"``), all the prefixes
    (``kind="prefix"``) or every range (``kind="all"``). ``abs_error`` is the mean absolute
    error per query; ``rel_error`` divides it by the number of records.
    """

    title = "Range queries"
    task_kind = WORKLOAD
    column_kinds = (NUMERIC_COLUMN, CATEGORICAL)

    def __init__(
        self, column: str = "", bins: int = 32, n_queries: int = 100, kind: str = "random", seed: int = 0
    ) -> None:
        if kind not in ("random", "prefix", "all"):
            raise ValueError(f"kind must be 'random', 'prefix' or 'all', got {kind!r}")
        self.column, self.bins, self.n_queries, self.kind, self.seed = (
            column,
            int(bins),
            int(n_queries),
            kind,
            int(seed),
        )

    def ranges(self, cells: int) -> list[tuple[int, int]]:
        if self.kind == "prefix":
            return [(0, j) for j in range(1, cells + 1)]
        if self.kind == "all":
            return [(i, j) for i in range(cells) for j in range(i + 1, cells + 1)]
        rng = np.random.default_rng(self.seed)
        out = []
        for _ in range(self.n_queries):
            a, b = sorted(rng.choice(cells + 1, size=2, replace=False))
            out.append((int(a), int(b)))
        return out

    def tasks(self, table: Table) -> list[Task]:
        codes, labels = _cells(table, self.column, self.bins)
        rows = []
        for a, b in self.ranges(len(labels)):
            row = np.zeros(len(labels))
            row[a:b] = 1.0
            rows.append(row)
        workload = np.array(rows)
        return [WorkloadTask(_counts(codes, len(labels)), workload, labels)]

    def errors(self, estimate: Any, exact: Any, table: Table, tasks: list[Task]) -> dict[str, float]:
        difference = np.abs(np.asarray(estimate) - np.asarray(exact))
        mean = float(difference.mean())
        return {"abs_error": mean, "rel_error": mean / max(table.n, 1), "max_error": float(difference.max())}

    def describe(self) -> str:
        what = {"random": f"{self.n_queries} random ranges", "prefix": "all prefixes", "all": "all ranges"}[self.kind]
        return f"Range queries on {self.column} ({what})"


@register_query("quantile")
class Quantile(Query):
    """The q-quantile of a numeric column (q = 0.5: the median). ``rel_error`` is the rank error of the answer.

    The rank error is the fraction of the records that lie between the answer and the true
    quantile: 0 when the answer is a q-quantile of the data or falls in the gap next to it
    (a median of 45.5 for integer ages whose median is 45), 0.1 when 10% of the records
    separate them.
    """

    title = "Quantile"
    task_kind = QUANTILE

    def __init__(
        self, column: str = "", q: float = 0.5, lower: float | None = None, upper: float | None = None
    ) -> None:
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0, 1), got {q}")
        self.column, self.q, self.lower, self.upper = column, float(q), lower, upper

    def tasks(self, table: Table) -> list[Task]:
        low, high = _bounds(table.column(self.column), self.lower, self.upper)
        return [QuantileTask(numeric_values(table.frame, table.column(self.column)), low, high, self.q)]

    def errors(self, estimate: Any, exact: Any, table: Table, tasks: list[Task]) -> dict[str, float]:
        values, n = tasks[0].values, max(len(tasks[0].values), 1)
        x, t = float(estimate), float(exact)
        below, at_most = np.searchsorted(values, x, side="left"), np.searchsorted(values, x, side="right")
        if below / n <= self.q <= at_most / n:  # x is itself a q-quantile of the data
            between = 0
        elif x > t:  # records in (t, x]
            between = at_most - np.searchsorted(values, t, side="right")
        else:  # records in [x, t)
            between = np.searchsorted(values, t, side="left") - below
        return {"abs_error": abs(x - t), "rel_error": max(int(between), 0) / n}

    def describe(self) -> str:
        return "Median of " + self.column if self.q == 0.5 else f"{self.q:g}-quantile of {self.column}"


@register_query("mode")
class Mode(Query):
    """The most frequent category of a column: a selection. ``correct`` = 1 if the true mode is returned.

    ``abs_error`` is the regret: how many records fewer the returned category has than
    the true mode; ``rel_error`` divides it by the count of the true mode.
    """

    title = "Mode (most frequent value)"
    task_kind = SELECTION
    column_kinds = (CATEGORICAL, NUMERIC_COLUMN)

    def __init__(self, column: str = "", bins: int = 10) -> None:
        self.column, self.bins = column, int(bins)

    def tasks(self, table: Table) -> list[Task]:
        codes, labels = _cells(table, self.column, self.bins)
        return [SelectionTask(_counts(codes, len(labels)), 1.0, labels, monotone=True)]

    def errors(self, estimate: Any, exact: Any, table: Table, tasks: list[Task]) -> dict[str, float]:
        scores = tasks[0].scores
        regret = float(scores.max() - scores[int(estimate)])
        return {
            "abs_error": regret,
            "rel_error": regret / max(float(scores.max()), 1.0),
            "correct": float(regret == 0),
        }

    def describe(self) -> str:
        return f"Mode of {self.column}"
