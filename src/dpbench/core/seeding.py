"""Seeding helpers: every source of randomness of a run derives from the run's seed."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def derive_seed(*parts: object) -> int:
    """A deterministic 31-bit seed computed from arbitrary parts.

    Unlike ``hash()``, the value is the same in every Python process, so a run can be
    reproduced exactly: ``derive_seed(seed, "noise", mechanism, epsilon)``.
    """
    text = "|".join(repr(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch's global generators."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A dedicated PyTorch generator (sampling, noise...) independent of the global one."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def resolve_device(device: str) -> torch.device:
    """``"auto"`` selects CUDA when available, otherwise the CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
