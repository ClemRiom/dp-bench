from __future__ import annotations

from dpbench.core.registry import register_mechanism
from dpbench.mechanisms.dpsgd import DPSGD


@register_mechanism("dp_lap_sgd")
class DPLapSGD(DPSGD):
    """DP-SGD with per-example L1 clipping and Laplace noise, accounted with the PLD accountant."""

    title = "DP-LAP-SGD"
    noise = "laplace"
    clipping_norm = 1
    default_accountant = "pld"
