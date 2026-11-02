from __future__ import annotations

from dpbench.core.registry import register_mechanism
from dpbench.mechanisms.dpsgd import DPSGD


@register_mechanism("dp_adam")
class DPAdam(DPSGD):
    """DP-Adam: the private gradient of DP-SGD fed to Adam (post-processing, so the same privacy guarantee)."""

    title = "DP-Adam"
    default_optimizer = "adam"
    default_lr = 1e-3
