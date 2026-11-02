"""Built-in accountants.

How the privacy loss of T noisy steps adds up, from the loosest to the tightest:

* ``basic``: basic composition, the epsilons of the steps are summed;
* ``advanced``: advanced composition, they grow like the square root of T;
* ``rdp``: Rényi DP, the Rényi divergences of the steps are summed exactly, then converted;
* ``prv`` and ``pld``: the privacy loss distributions of the steps are composed numerically
  (a convolution), which is almost exact;
* ``gdp``: Gaussian DP with a central limit theorem, an approximation, not a guarantee.

Every accountant but GDP returns a guaranteed upper bound on epsilon.
"""

from __future__ import annotations

import functools
import math
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from dpbench.accounting.base import Accountant
from dpbench.core.registry import register_accountant

RDP_ORDERS = tuple(
    float(order)
    for order in np.concatenate(
        [
            [1.01, 1.02, 1.05],
            1 + np.arange(1, 100) / 10,
            np.arange(11, 64),
            [64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024],
        ]
    )
)


def _dp_event(noise_multiplier: float, sample_rate: float, noise: str):
    from dp_accounting import dp_event

    inner = (
        dp_event.GaussianDpEvent(noise_multiplier) if noise == "gaussian" else dp_event.LaplaceDpEvent(noise_multiplier)
    )
    return inner if sample_rate >= 1 else dp_event.PoissonSampledDpEvent(sample_rate, inner)


@register_accountant("rdp")
class RDPAccountant(Accountant):
    """Rényi DP accountant (Mironov 2017; conversion to (ε, δ) of Canonne, Kamath & Steinke 2020): an upper bound, slightly loose.

    Orders from 1.01 to 1024. Implementation: Google's ``dp_accounting``.
    """

    def __init__(self, orders: list[float] | None = None) -> None:
        self.orders = tuple(orders) if orders is not None else RDP_ORDERS

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        from dp_accounting.rdp import rdp_privacy_accountant

        accountant = rdp_privacy_accountant.RdpAccountant(orders=list(self.orders))
        accountant.compose(_dp_event(noise_multiplier, sample_rate, noise), steps)
        return accountant.get_epsilon(delta)


@register_accountant("prv")
class PRVAccountant(Accountant):
    """Privacy random variables accountant (Gopi, Lee & Wutschitz 2021): numerical composition, returns a certified upper bound.

    Implementation: Opacus. ``eps_error`` is the tolerated discretisation error on ε.
    """

    def __init__(self, eps_error: float = 0.01) -> None:
        self.eps_error = eps_error

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        from opacus.accountants import PRVAccountant as OpacusPRV

        accountant = OpacusPRV()
        accountant.history = [(noise_multiplier, sample_rate, steps)]
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)  # Opacus' RDP domain estimate warns on extreme orders
            return accountant.get_epsilon(delta, eps_error=self.eps_error)


@register_accountant("pld")
class PLDAccountant(Accountant):
    """Privacy loss distribution accountant (Doroshenko et al. 2022, connect-the-dots): tight upper bound, Gaussian and Laplace noise.

    Implementation: Google's ``dp_accounting``; ``value_discretization_interval`` sets the
    discretisation of the privacy loss.
    """

    supported_noises = ("gaussian", "laplace")

    def __init__(self, value_discretization_interval: float = 1e-3) -> None:
        self.value_discretization_interval = value_discretization_interval

    #: Below this Gaussian noise multiplier the privacy loss is too spread out to be discretised in a
    #: reasonable time (minutes); the RDP bound, also guaranteed, is returned (ε is huge there anyway).
    SMALL_GAUSSIAN_NOISE = 0.3

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        from dp_accounting.pld import pld_privacy_accountant

        if noise == "gaussian" and noise_multiplier < self.SMALL_GAUSSIAN_NOISE:
            return RDPAccountant().epsilon(noise_multiplier, sample_rate, steps, delta, noise)

        accountant = pld_privacy_accountant.PLDAccountant(
            value_discretization_interval=self.value_discretization_interval
        )
        accountant.compose(_dp_event(noise_multiplier, sample_rate, noise), steps)
        return accountant.get_epsilon(delta)


def gdp_delta(epsilon: float, mu: float) -> float:
    """delta(epsilon) of mu-GDP: Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2)."""
    first = ndtr(-epsilon / mu + mu / 2)
    second = math.exp(epsilon + log_ndtr(-epsilon / mu - mu / 2))
    return float(first - second)


def gdp_epsilon(mu: float, delta: float) -> float:
    """Smallest epsilon such that mu-GDP implies (epsilon, delta)-DP."""
    if not math.isfinite(mu):
        return math.inf
    if gdp_delta(0.0, mu) <= delta:
        return 0.0
    high = 1.0
    while gdp_delta(high, mu) > delta:
        high *= 2
        if high > 1e6:
            return math.inf
    return float(brentq(lambda e: gdp_delta(e, mu) - delta, 0.0, high, xtol=1e-12, rtol=1e-10))


@register_accountant("gdp")
class GDPAccountant(Accountant):
    """Gaussian DP accountant (Dong, Roth & Su 2022) with the central limit theorem of Bu et al. (2020): an approximation, not a guaranteed bound.

    mu = q * sqrt(T * (exp(1/sigma^2) - 1)) for Poisson sampling, and the exact
    mu = sqrt(T) / sigma without sampling (q = 1).
    """

    approximate = True

    @staticmethod
    def mu(noise_multiplier: float, sample_rate: float, steps: int) -> float:
        if sample_rate >= 1:
            return math.sqrt(steps) / noise_multiplier
        exponent = 1.0 / noise_multiplier**2
        if exponent > 700:
            return math.inf
        return sample_rate * math.sqrt(steps * math.expm1(exponent))

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        return gdp_epsilon(self.mu(noise_multiplier, sample_rate, steps), delta)


@functools.lru_cache(maxsize=4096)
def step_epsilon(noise_multiplier: float, sample_rate: float, delta: float, noise: str = "gaussian") -> float:
    """Epsilon of ONE Poisson-subsampled noisy step at the given delta (tight, with the PLD accountant).

    The building block of the composition theorems: each step is (step_epsilon, delta)-DP.
    The discretisation of the privacy loss (1e-3 by default) is refined for small epsilons,
    which a coarse grid would round up to the grid step.
    """
    if noise == "laplace" and noise_multiplier < PLDAccountant.SMALL_GAUSSIAN_NOISE:
        # Pure ε-DP: Laplace of scale σC on a sum of sensitivity C is (1/σ)-DP, and Poisson subsampling with
        # rate q gives log(1 + q (e^(1/σ) - 1)) (amplification by subsampling).
        return math.log1p(sample_rate * math.expm1(min(1 / noise_multiplier, 700.0)))
    coarse = PLDAccountant().epsilon(noise_multiplier, sample_rate, 1, delta, noise)
    if coarse >= 0.1:
        return coarse
    interval = max(coarse / 100, 1e-7)
    return PLDAccountant(value_discretization_interval=interval).epsilon(noise_multiplier, sample_rate, 1, delta, noise)


@register_accountant("basic")
class BasicCompositionAccountant(Accountant):
    """Basic composition (Dwork, Kenthapadi, McSherry, Mironov & Naor 2006): T steps that are each (ε₀, δ/T)-DP are (T·ε₀, δ)-DP.

    The simplest rule: the epsilons add up. A guaranteed upper bound, but very loose for the
    thousands of steps of a training. Gaussian and Laplace noise.
    """

    supported_noises = ("gaussian", "laplace")

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        return steps * step_epsilon(noise_multiplier, sample_rate, delta / steps, noise)


@register_accountant("advanced")
class AdvancedCompositionAccountant(Accountant):
    """Advanced composition (Dwork, Rothblum & Vadhan 2010): ε = ε₀√(2T ln(1/δ′)) + T·ε₀(e^ε₀ − 1), at δ = T·δ₀ + δ′.

    Half of δ goes to the steps (δ₀ = δ / 2T each), the other half is the slack δ′. The
    total grows like √T instead of T. Never worse than basic composition (the smaller of
    the two bounds is returned; both hold at δ). Gaussian and Laplace noise.
    """

    supported_noises = ("gaussian", "laplace")

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        basic = steps * step_epsilon(noise_multiplier, sample_rate, delta / steps, noise)
        eps0 = step_epsilon(noise_multiplier, sample_rate, delta / (2 * steps), noise)
        if eps0 > 1:  # e^ε₀ - 1 > ε₀: the advanced bound is then worse than the basic one
            return basic
        advanced = eps0 * math.sqrt(2 * steps * math.log(2 / delta)) + steps * eps0 * math.expm1(eps0)
        return min(advanced, basic)
