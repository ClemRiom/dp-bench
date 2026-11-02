"""What a query asks a mechanism to release: four kinds of *tasks*.

A query (``queries.py``) reads a table and turns itself into one or more tasks; a
mechanism (``mechanisms.py``) releases a task with differential privacy. The task carries
everything the mechanism needs -- the exact answer and its sensitivity -- and nothing
about the table, so a mechanism works for every query of the same kind.

The neighbouring relation is *add or remove one record*: the sensitivities below are the
largest change that one record can cause.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

NUMERIC = "numeric"
SELECTION = "selection"
QUANTILE = "quantile"
WORKLOAD = "workload"
TASK_KINDS = (NUMERIC, SELECTION, QUANTILE, WORKLOAD)


@dataclass
class NumericTask:
    """A vector of numbers (counts, sums...) to release with additive noise.

    ``l1_sensitivity`` and ``l2_sensitivity`` bound the L1 and L2 norms of the change of
    ``answer`` when one record is added or removed. ``integer`` is true when the answer is
    integer-valued with an integer sensitivity (counts), which discrete mechanisms need.
    """

    answer: np.ndarray
    l1_sensitivity: float
    l2_sensitivity: float
    integer: bool = False
    labels: list[str] | None = None
    kind: ClassVar[str] = NUMERIC

    def __post_init__(self) -> None:
        self.answer = np.atleast_1d(np.asarray(self.answer, dtype=np.float64))


@dataclass
class SelectionTask:
    """Pick the candidate with the highest score (e.g. the most frequent category).

    ``sensitivity`` bounds the change of every score; ``monotone`` is true when adding a
    record can only increase the scores (counts), which lets the selection mechanisms use
    half the noise.
    """

    scores: np.ndarray
    sensitivity: float
    candidates: list[str]
    monotone: bool = True
    kind: ClassVar[str] = SELECTION

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64)


@dataclass
class QuantileTask:
    """The ``q``-quantile of numeric values known to lie in ``[lower, upper]``."""

    values: np.ndarray
    lower: float
    upper: float
    q: float = 0.5
    kind: ClassVar[str] = QUANTILE

    def __post_init__(self) -> None:
        self.values = np.sort(np.clip(np.asarray(self.values, dtype=np.float64), self.lower, self.upper))


@dataclass
class WorkloadTask:
    """Many linear queries at once: ``workload @ histogram``, e.g. counts of records in many ranges.

    ``histogram`` holds the number of records in each cell of a discretised domain and
    ``workload`` has one row per query (1 for the cells the query counts, 0 elsewhere).
    One record changes one cell of the histogram by 1.
    """

    histogram: np.ndarray
    workload: np.ndarray
    cell_labels: list[str] | None = None
    kind: ClassVar[str] = WORKLOAD

    def __post_init__(self) -> None:
        self.histogram = np.asarray(self.histogram, dtype=np.float64)
        self.workload = np.atleast_2d(np.asarray(self.workload, dtype=np.float64))
        if self.workload.shape[1] != self.histogram.shape[0]:
            raise ValueError("the workload must have one column per cell of the histogram")

    @property
    def answer(self) -> np.ndarray:
        return self.workload @ self.histogram

    @property
    def l1_sensitivity(self) -> float:
        """Sensitivity of answering every query directly: the largest number of queries that count one cell."""
        return float(np.abs(self.workload).sum(axis=0).max())


Task = NumericTask | SelectionTask | QuantileTask | WorkloadTask
