"""Query mechanisms: release a task (``tasks.py``) with differential privacy.

A mechanism declares the kinds of tasks it supports and implements :meth:`release`. The
four built-in ones:

* **Laplace** and **Gaussian** (analytic calibration): additive noise for numeric answers
  (counts, sums, histograms...) and for workloads answered query by query;
* **polar Laplace**: a radial noise and a directional noise (the classic counterpart of
  GeoDP's idea, with a proved guarantee);
* **exponential mechanism**: selection (the most frequent category) and quantiles.

All mechanisms are calibrated for the *add or remove one record* neighbouring relation.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from scipy.optimize import brentq

from dpbench.accounting.accountants import gdp_delta
from dpbench.classic.tasks import NUMERIC, QUANTILE, SELECTION, WORKLOAD, Task
from dpbench.core.registry import register_query_mechanism


class QueryMechanism(ABC):
    """Base class of the query mechanism plugins."""

    name: ClassVar[str]
    title: ClassVar[str] = ""
    #: Kinds of tasks this mechanism can release.
    supports: ClassVar[tuple[str, ...]] = (NUMERIC,)
    #: True for an (epsilon, delta)-DP mechanism (it needs delta > 0), False for a pure epsilon-DP one.
    uses_delta: ClassVar[bool] = False

    @abstractmethod
    def release(self, task: Task, epsilon: float, delta: float, rng: np.random.Generator) -> Any:
        """A differentially private version of the task's answer."""

    def guarantee(self, epsilon: float, delta: float) -> tuple[float, float]:
        """The (epsilon, delta) actually guaranteed when called with this budget."""
        return epsilon, (delta if self.uses_delta else 0.0)

    def check(self, task: Task, epsilon: float, delta: float) -> None:
        if task.kind not in self.supports:
            raise ValueError(f"{self.name} cannot release {task.kind} tasks (it supports: {', '.join(self.supports)})")
        if not epsilon > 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if self.uses_delta and not 0 < delta < 1:
            raise ValueError(f"{self.name} is (epsilon, delta)-DP and needs 0 < delta < 1, got {delta}")


def numeric_view(task: Task) -> tuple[np.ndarray, float, float, bool]:
    """(answer, L1 sensitivity, L2 sensitivity, integer) of a numeric task, or of a workload answered query by query.

    One record changes one cell of a workload's histogram, hence every query that counts
    this cell: the L1 (L2) sensitivity is the largest L1 (L2) norm of a column of the workload.
    """
    if task.kind == NUMERIC:
        return task.answer, float(task.l1_sensitivity), float(task.l2_sensitivity), bool(task.integer)
    if task.kind == WORKLOAD:
        columns = np.abs(task.workload)
        integer = bool(np.all(task.workload == np.round(task.workload)))
        return task.answer, float(columns.sum(axis=0).max()), float(np.sqrt((columns**2).sum(axis=0)).max()), integer
    raise ValueError(f"not a numeric task: {task.kind}")


@register_query_mechanism("laplace")
class Laplace(QueryMechanism):
    """Laplace mechanism (Dwork, McSherry, Nissim & Smith 2006): noise Lap(Δ₁/ε) on every coordinate. Pure ε-DP."""

    title = "Laplace"
    supports = (NUMERIC, WORKLOAD)

    def release(self, task, epsilon, delta, rng):
        answer, l1, _, _ = numeric_view(task)
        return answer + rng.laplace(0.0, l1 / epsilon, answer.shape)


@functools.lru_cache(maxsize=4096)
def _analytic_mu(epsilon: float, delta: float) -> float:
    """Largest μ such that the Gaussian mechanism with Δ₂/σ = μ is (ε, δ)-DP (exact profile)."""
    high = 1.0
    while gdp_delta(epsilon, high) < delta:
        high *= 2
        if high > 1e6:
            return high
    low = high / 2
    while gdp_delta(epsilon, low) > delta:
        low /= 2
        if low < 1e-12:
            return low
    return float(brentq(lambda mu: gdp_delta(epsilon, mu) - delta, low, high, xtol=1e-14, rtol=1e-12))


def analytic_gaussian_sigma(epsilon: float, delta: float, l2_sensitivity: float) -> float:
    """Smallest σ for which the Gaussian mechanism is (ε, δ)-DP (Balle & Wang 2018), for any ε > 0."""
    return l2_sensitivity / _analytic_mu(float(epsilon), float(delta))


@register_query_mechanism("gaussian")
class Gaussian(QueryMechanism):
    """Gaussian mechanism, noise N(0, σ²) with the exact (analytic) calibration of Balle & Wang (2018): (ε, δ)-DP.

    σ is the smallest standard deviation whose exact privacy profile gives δ at this ε
    (formula ``gdp_delta`` with μ = Δ₂/σ). Unlike the textbook σ = Δ₂√(2 ln(1.25/δ))/ε,
    it is valid for every ε and always adds less noise.
    """

    title = "Gaussian"
    supports = (NUMERIC, WORKLOAD)
    uses_delta = True

    def release(self, task, epsilon, delta, rng):
        answer, _, l2, _ = numeric_view(task)
        return answer + rng.normal(0.0, analytic_gaussian_sigma(epsilon, delta, l2), answer.shape)


@register_query_mechanism("polar_laplace")
class PolarLaplace(QueryMechanism):
    """Polar Laplace mechanism: a radial noise and a directional noise (L2 K-norm mechanism). Pure ε-DP.

    The noise is ``R * U``: a direction ``U`` drawn uniformly on the unit sphere, and a
    radius ``R`` drawn from a Gamma(k, Δ₂/ε) distribution (k = dimension). Its density is
    proportional to exp(-ε‖z‖₂/Δ₂), so by the triangle inequality it is ε-DP for the L2
    sensitivity (Hardt & Talwar 2010). In two dimensions it is the planar Laplace mechanism
    of geo-indistinguishability (Andrés et al. 2013). In one dimension it is the Laplace
    mechanism.
    """

    title = "Polar Laplace (radial + directional)"
    supports = (NUMERIC, WORKLOAD)

    def release(self, task, epsilon, delta, rng):
        answer, _, l2, _ = numeric_view(task)
        direction = rng.normal(size=answer.shape)
        direction /= np.linalg.norm(direction)
        radius = rng.gamma(shape=answer.size, scale=l2 / epsilon)
        return answer + radius * direction


@register_query_mechanism("exponential")
class Exponential(QueryMechanism):
    """Exponential mechanism (McSherry & Talwar 2007): candidate r with probability ∝ exp(ε·score_r/(2Δ)). Pure ε-DP.

    For a quantile, the candidates are the intervals between consecutive data points,
    weighted by their length (Smith 2011), and the answer is uniform in the chosen interval.
    For monotone scores (counts), ε/Δ replaces ε/(2Δ).
    """

    title = "Exponential mechanism"
    supports = (SELECTION, QUANTILE)

    def release(self, task, epsilon, delta, rng):
        if task.kind == SELECTION:
            scale = epsilon / (task.sensitivity if task.monotone else 2 * task.sensitivity)
            logits = scale * task.scores
            return int(np.argmax(logits + rng.gumbel(size=logits.shape)))  # Gumbel-max trick = sampling ∝ exp(logits)
        values, n = task.values, len(task.values)
        points = np.concatenate([[task.lower], values, [task.upper]])
        widths = np.diff(points)
        utility = -np.abs(np.arange(n + 1) - task.q * n)  # rank error of the interval; sensitivity 1
        with np.errstate(divide="ignore"):
            logits = np.log(widths) + epsilon * utility / 2.0
        i = int(np.argmax(logits + rng.gumbel(size=logits.shape)))
        return float(rng.uniform(points[i], points[i + 1]))
