"""Query mechanisms: release a task (``tasks.py``) with differential privacy.

A mechanism declares the kinds of tasks it supports and implements :meth:`release`. The
built-in ones:

* **additive noise** (numeric answers, and workloads answered query by query): Laplace,
  Gaussian (analytic and classic calibration), discrete Laplace, polar Laplace (radial +
  directional noise);
* **selection** (the most frequent category): exponential mechanism, report noisy max,
  permute-and-flip;
* **quantiles**: exponential mechanism over the intervals between the data points, or
  inversion of a noisy histogram;
* **workloads of range queries**: identity (noisy histogram), uniform, hierarchical tree
  with constrained inference, MWEM.

A new mechanism is either a subclass of :class:`QueryMechanism` registered with
``@register_query_mechanism("name")``, or, for additive noise, a plain function registered
with ``@noise_mechanism("name")`` (see :func:`noise_mechanism`).

All mechanisms are calibrated for the *add or remove one record* neighbouring relation.
"""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
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


def classic_gaussian_sigma(epsilon: float, delta: float, l2_sensitivity: float) -> float:
    """σ = Δ₂ √(2 ln(1.25/δ)) / ε (Dwork & Roth, Theorem A.1), proved for ε < 1 only."""
    if epsilon >= 1:
        raise ValueError(
            f"the classic Gaussian calibration is only proved for epsilon < 1 (got {epsilon}); "
            "use the analytic Gaussian ('gaussian') for larger budgets"
        )
    return l2_sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


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

    title = "Gaussian (analytic σ)"
    supports = (NUMERIC, WORKLOAD)
    uses_delta = True

    def release(self, task, epsilon, delta, rng):
        answer, _, l2, _ = numeric_view(task)
        return answer + rng.normal(0.0, analytic_gaussian_sigma(epsilon, delta, l2), answer.shape)


@register_query_mechanism("gaussian_classic")
class ClassicGaussian(QueryMechanism):
    """Gaussian mechanism with the classic calibration σ = Δ₂√(2 ln(1.25/δ))/ε: (ε, δ)-DP, proved for ε < 1 only."""

    title = "Gaussian (classic σ)"
    supports = (NUMERIC, WORKLOAD)
    uses_delta = True

    def check(self, task, epsilon, delta):
        super().check(task, epsilon, delta)
        classic_gaussian_sigma(epsilon, delta, 1.0)  # raises for epsilon >= 1: the run is then marked "skipped"

    def release(self, task, epsilon, delta, rng):
        answer, _, l2, _ = numeric_view(task)
        return answer + rng.normal(0.0, classic_gaussian_sigma(epsilon, delta, l2), answer.shape)


@register_query_mechanism("discrete_laplace")
class DiscreteLaplace(QueryMechanism):
    """Discrete Laplace (geometric) mechanism (Ghosh, Roughgarden & Sundararajan 2009): integer noise for counts. Pure ε-DP.

    P(noise = k) is proportional to exp(-ε|k|/Δ₁); the noise is the difference of two
    geometric variables. The released counts stay integers.
    """

    title = "Discrete Laplace"
    supports = (NUMERIC, WORKLOAD)

    def check(self, task, epsilon, delta):
        super().check(task, epsilon, delta)
        _, l1, _, integer = numeric_view(task)
        if not integer or l1 != round(l1):
            raise ValueError("the discrete Laplace mechanism needs integer answers with an integer sensitivity")

    def release(self, task, epsilon, delta, rng):
        answer, l1, _, _ = numeric_view(task)
        p = -math.expm1(-epsilon / l1)  # 1 - exp(-ε/Δ)
        return answer + (rng.geometric(p, answer.shape) - rng.geometric(p, answer.shape))


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


def _selection_scale(task, epsilon: float) -> float:
    """ε/Δ for monotone scores (counts), ε/(2Δ) in general."""
    return epsilon / (task.sensitivity if task.monotone else 2 * task.sensitivity)


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
            logits = _selection_scale(task, epsilon) * task.scores
            return int(np.argmax(logits + rng.gumbel(size=logits.shape)))  # Gumbel-max trick = sampling ∝ exp(logits)
        values, n = task.values, len(task.values)
        points = np.concatenate([[task.lower], values, [task.upper]])
        widths = np.diff(points)
        utility = -np.abs(np.arange(n + 1) - task.q * n)  # rank error of the interval; sensitivity 1
        with np.errstate(divide="ignore"):
            logits = np.log(widths) + epsilon * utility / 2.0
        i = int(np.argmax(logits + rng.gumbel(size=logits.shape)))
        return float(rng.uniform(points[i], points[i + 1]))


@register_query_mechanism("report_noisy_max")
class ReportNoisyMax(QueryMechanism):
    """Report noisy max (Dwork & Roth, Claim 3.9): add Lap(Δ/ε) to every count and return the largest. Pure ε-DP."""

    title = "Report noisy max"
    supports = (SELECTION,)

    def release(self, task, epsilon, delta, rng):
        scale = 1.0 / _selection_scale(task, epsilon)
        return int(np.argmax(task.scores + rng.laplace(0.0, scale, task.scores.shape)))


@register_query_mechanism("permute_and_flip")
class PermuteAndFlip(QueryMechanism):
    """Permute-and-flip (McKenna & Sheldon 2020): never worse than the exponential mechanism. Pure ε-DP.

    Visit the candidates in a random order and accept candidate r with probability
    exp(ε(score_r - max score)/(2Δ)); the best candidate is always accepted.
    """

    title = "Permute-and-flip"
    supports = (SELECTION,)

    def release(self, task, epsilon, delta, rng):
        scale = _selection_scale(task, epsilon)
        best = task.scores.max()
        for r in rng.permutation(len(task.scores)):
            if rng.random() <= math.exp(scale * (task.scores[r] - best)):
                return int(r)
        return int(np.argmax(task.scores))  # unreachable: the best candidate has probability 1


@register_query_mechanism("histogram_quantile")
class HistogramQuantile(QueryMechanism):
    """Quantile read on a noisy histogram (Laplace, ``bins`` equal bins of [lower, upper]). Pure ε-DP."""

    title = "Noisy histogram quantile"
    supports = (QUANTILE,)

    def __init__(self, bins: int = 32) -> None:
        self.bins = int(bins)

    def release(self, task, epsilon, delta, rng):
        edges = np.linspace(task.lower, task.upper, self.bins + 1)
        counts = np.histogram(task.values, bins=edges)[0].astype(np.float64)
        noisy = np.clip(counts + rng.laplace(0.0, 1.0 / epsilon, counts.shape), 0.0, None)
        if noisy.sum() <= 0:
            return float((task.lower + task.upper) / 2)
        cumulative = np.cumsum(noisy) / noisy.sum()
        i = int(np.searchsorted(cumulative, task.q))
        before = cumulative[i - 1] if i > 0 else 0.0
        fraction = (task.q - before) / max(cumulative[i] - before, 1e-12)
        return float(edges[i] + fraction * (edges[i + 1] - edges[i]))


@register_query_mechanism("identity")
class Identity(QueryMechanism):
    """Identity strategy: a noisy histogram (Lap(1/ε) per cell), then every range query summed from it. Pure ε-DP."""

    title = "Identity (noisy histogram)"
    supports = (WORKLOAD,)

    def release(self, task, epsilon, delta, rng):
        noisy = task.histogram + rng.laplace(0.0, 1.0 / epsilon, task.histogram.shape)
        return task.workload @ noisy


@register_query_mechanism("uniform")
class Uniform(QueryMechanism):
    """Uniform baseline: only a noisy total count, spread evenly over the cells. Pure ε-DP; a sanity floor."""

    title = "Uniform (noisy total only)"
    supports = (WORKLOAD,)

    def release(self, task, epsilon, delta, rng):
        total = max(task.histogram.sum() + rng.laplace(0.0, 1.0 / epsilon), 0.0)
        return task.workload @ np.full(task.histogram.shape, total / len(task.histogram))


@register_query_mechanism("hierarchical")
class Hierarchical(QueryMechanism):
    """Hierarchical mechanism (Hay, Rastogi, Miklau & Suciu 2010): noisy counts of a tree of ranges, made consistent. Pure ε-DP.

    The cells are the leaves of a ``branching``-ary tree; every node counts its leaves.
    One record is in one node per level, so the sensitivity is the number of levels L and
    every node gets Lap(L/ε) noise. Constrained inference (least squares under the
    constraint "a parent is the sum of its children") then combines the noisy counts: a
    range is answered from O(log m) nodes instead of up to m cells.
    """

    title = "Hierarchical (tree)"
    supports = (WORKLOAD,)

    def __init__(self, branching: int = 2) -> None:
        if branching < 2:
            raise ValueError("branching must be at least 2")
        self.branching = int(branching)

    def release(self, task, epsilon, delta, rng):
        b, m = self.branching, len(task.histogram)
        height = max(1, math.ceil(math.log(m, b) - 1e-9)) if m > 1 else 1
        leaves = np.zeros(b**height)
        leaves[:m] = task.histogram
        levels = [leaves]  # levels[0] = leaves, levels[-1] = root
        while len(levels[-1]) > 1:
            levels.append(levels[-1].reshape(-1, b).sum(axis=1))
        scale = len(levels) / epsilon
        noisy = [level + rng.laplace(0.0, scale, level.shape) for level in levels]
        # Bottom-up pass: z combines a node's own noisy count with the sum of its children's z.
        z = [noisy[0]]
        for height_l in range(2, len(levels) + 1):  # leaves have height 1
            children = z[-1].reshape(-1, b).sum(axis=1)
            w = (b**height_l - b ** (height_l - 1)) / (b**height_l - 1)
            z.append(w * noisy[height_l - 1] + (1 - w) * children)
        # Top-down pass: spread each parent's residual equally over its children.
        estimate = z[-1]
        for level in range(len(levels) - 2, -1, -1):
            children = z[level].reshape(-1, b)
            residual = (estimate - children.sum(axis=1)) / b
            estimate = (children + residual[:, None]).ravel()
        return task.workload @ estimate[:m]


@register_query_mechanism("mwem")
class MWEM(QueryMechanism):
    """MWEM (Hardt, Ligett & McSherry 2012): a synthetic histogram improved by multiplicative weights. Pure ε-DP.

    10% of ε gives a noisy total; the rest is split over ``rounds`` rounds. Each round
    selects the worst-answered query with the exponential mechanism (half of the round's
    budget), measures it with Laplace noise (the other half), and updates the synthetic
    histogram with multiplicative weights. By basic composition the total is ε.
    """

    title = "MWEM"
    supports = (WORKLOAD,)

    def __init__(self, rounds: int = 10, total_fraction: float = 0.1) -> None:
        if rounds < 1 or not 0 < total_fraction < 1:
            raise ValueError("rounds must be >= 1 and total_fraction in (0, 1)")
        self.rounds = int(rounds)
        self.total_fraction = float(total_fraction)

    def release(self, task, epsilon, delta, rng):
        workload, histogram = task.workload, task.histogram
        sensitivity = float(np.abs(workload).max())  # one record changes each query by at most this (1 for ranges)
        total = max(histogram.sum() + rng.laplace(0.0, 1.0 / (self.total_fraction * epsilon)), 1.0)
        per_round = (1 - self.total_fraction) * epsilon / self.rounds
        synthetic = np.full(histogram.shape, total / len(histogram))
        exact = workload @ histogram
        measured: list[tuple[int, float]] = []
        for _ in range(self.rounds):
            # Selection with ε/2 of the round: the score |error| is not monotone, so the factor is (ε/2)/(2Δ).
            scores = np.abs(workload @ synthetic - exact)
            choice = int(np.argmax(per_round / 2 / (2 * sensitivity) * scores + rng.gumbel(size=scores.shape)))
            # Measurement with the other ε/2: Laplace noise of scale Δ/(ε/2).
            measured.append((choice, float(exact[choice] + rng.laplace(0.0, sensitivity / (per_round / 2)))))
            for _ in range(10):  # a few passes of multiplicative weights over all the measurements
                for query, value in measured:
                    row = workload[query]
                    synthetic = synthetic * np.exp(row * (value - row @ synthetic) / (2 * total))
                    synthetic *= total / synthetic.sum()
        return workload @ synthetic


NoiseFunction = Callable[[np.ndarray, float, float, float, np.random.Generator], Any]


def noise_mechanism(
    name: str,
    *,
    title: str = "",
    sensitivity: str = "l1",
    uses_delta: bool = False,
) -> Callable[[NoiseFunction], NoiseFunction]:
    """Register a plain function as an additive-noise query mechanism.

    The function receives the exact answer (a float64 vector), the sensitivity of the task
    (L1 or L2, as chosen with ``sensitivity``), epsilon, delta and a NumPy generator, and
    returns the noisy answer, of the same shape::

        @noise_mechanism("my_laplace", title="My Laplace")
        def my_laplace(answer, sensitivity, epsilon, delta, rng):
            return answer + rng.laplace(0.0, sensitivity / epsilon, size=answer.shape)

    The mechanism answers every numeric query (counts, sums, means, histograms,
    centroids...) and range queries answered one by one. Set ``uses_delta=True`` for an
    (epsilon, delta)-DP mechanism such as a Gaussian one.
    """
    if sensitivity not in ("l1", "l2"):
        raise ValueError(f"sensitivity must be 'l1' or 'l2', got {sensitivity!r}")

    def decorator(function: NoiseFunction) -> NoiseFunction:
        def release(self, task, epsilon, delta, rng):
            answer, l1, l2, _ = numeric_view(task)
            noisy = np.asarray(function(answer.copy(), l1 if sensitivity == "l1" else l2, epsilon, delta, rng))
            if noisy.shape != answer.shape:
                raise ValueError(f"{name} returned an answer of shape {noisy.shape} instead of {answer.shape}")
            return noisy.astype(np.float64)

        cls = type(
            f"NoiseMechanism_{name}",
            (QueryMechanism,),
            {
                "__doc__": function.__doc__ or f"Additive noise defined by the function {function.__name__}.",
                "__module__": function.__module__,
                "__qualname__": f"{function.__qualname__}.mechanism",
                "title": title or name,
                "supports": (NUMERIC, WORKLOAD),
                "uses_delta": uses_delta,
                "release": release,
                "function": staticmethod(function),
            },
        )
        register_query_mechanism(name)(cls)
        return function

    return decorator
