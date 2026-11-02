"""Privacy profile of the Gaussian mechanism, used to calibrate its noise exactly."""

from __future__ import annotations

import math

from scipy.special import log_ndtr, ndtr


def gdp_delta(epsilon: float, mu: float) -> float:
    """delta(epsilon) of mu-GDP: Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2)."""
    first = ndtr(-epsilon / mu + mu / 2)
    second = math.exp(epsilon + log_ndtr(-epsilon / mu - mu / 2))
    return float(first - second)
