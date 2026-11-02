"""Interface of the privacy accountants.

An accountant answers one question: after ``steps`` iterations of a mechanism that samples
each example with probability ``sample_rate`` (Poisson sampling) and adds noise of
multiplier ``noise_multiplier`` to a sum of contributions clipped to norm 1, what is the
smallest epsilon such that the whole training is (epsilon, delta)-DP?

The noise multiplier is the ratio between the noise scale and the sensitivity: Gaussian
noise of standard deviation ``noise_multiplier * C`` for an L2 clipping bound ``C``, or
Laplace noise of scale ``noise_multiplier * C`` for an L1 clipping bound ``C``. The
neighbouring relation is add-or-remove one example.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

NOISES = ("gaussian", "laplace")


class UnsupportedNoiseError(ValueError):
    """The accountant cannot analyse this kind of noise."""


class Accountant(ABC):
    """Base class of the accountant plugins."""

    name: ClassVar[str]
    supported_noises: ClassVar[tuple[str, ...]] = ("gaussian",)
    #: True when the result is an approximation rather than a guaranteed upper bound.
    approximate: ClassVar[bool] = False

    def check(self, noise: str) -> None:
        if noise not in self.supported_noises:
            raise UnsupportedNoiseError(
                f"accountant {self.name!r} does not support {noise} noise "
                f"(supported: {', '.join(self.supported_noises)})"
            )

    def epsilon(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
        delta: float,
        noise: str = "gaussian",
    ) -> float:
        """Epsilon spent after ``steps`` Poisson-subsampled noisy steps, at the given delta."""
        self.check(noise)
        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if steps <= 0:
            return 0.0
        if noise_multiplier <= 0:
            return math.inf
        value = float(self._epsilon(float(noise_multiplier), float(sample_rate), int(steps), float(delta), noise))
        if math.isnan(value):  # a NaN would silently pass every "epsilon <= target" check
            raise ArithmeticError(
                f"accountant {self.name!r} returned NaN (sigma={noise_multiplier}, q={sample_rate}, T={steps})"
            )
        return value

    def epsilons(
        self,
        noise_multiplier: float,
        sample_rate: float,
        steps: Sequence[int],
        delta: float,
        noise: str = "gaussian",
    ) -> list[float]:
        """Epsilon after each number of steps (the 'epsilon versus steps' curve)."""
        return [self.epsilon(noise_multiplier, sample_rate, s, delta, noise) for s in steps]

    @abstractmethod
    def _epsilon(self, noise_multiplier: float, sample_rate: float, steps: int, delta: float, noise: str) -> float:
        """Implementation for valid arguments (steps >= 1, noise_multiplier > 0)."""
