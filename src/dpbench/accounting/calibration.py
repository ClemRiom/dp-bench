"""Noise calibration: the smallest noise multiplier that keeps the training within a target epsilon."""

from __future__ import annotations

import math

from dpbench.accounting.base import Accountant


class CalibrationError(ValueError):
    """No noise multiplier in the search range reaches the target epsilon."""


def calibrate_noise(
    accountant: Accountant,
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
    noise: str = "gaussian",
    rtol: float = 1e-3,
    max_noise: float = 1e4,
) -> float:
    """Smallest noise multiplier sigma with ``accountant.epsilon(sigma, ...) <= target_epsilon``.

    Epsilon decreases when the noise grows, so a bisection on log(sigma) works. The returned
    value is always one for which the accountant was evaluated and certified the target
    (the upper end of the final bracket), and it is within a relative ``rtol`` of the exact
    threshold.
    """
    if math.isinf(target_epsilon):
        return 0.0
    if target_epsilon <= 0:
        raise CalibrationError(f"target epsilon must be positive, got {target_epsilon}")

    def epsilon(sigma: float) -> float:
        return accountant.epsilon(sigma, sample_rate, steps, delta, noise)

    high = 1.0
    while epsilon(high) > target_epsilon:
        high *= 2
        if high > max_noise:
            raise CalibrationError(
                f"epsilon = {target_epsilon} is out of reach with a noise multiplier below {max_noise} "
                f"(sample rate {sample_rate:.4g}, {steps} steps, delta = {delta:.3g})"
            )
    low = high / 2
    while epsilon(low) <= target_epsilon:
        high, low = low, low / 2
        if low < 1e-4:
            return high
    while high / low > 1 + rtol:
        middle = math.sqrt(low * high)
        if epsilon(middle) <= target_epsilon:
            high = middle
        else:
            low = middle
    return high
