"""Seeding helpers: every source of randomness of a benchmark derives from its seed."""

from __future__ import annotations

import hashlib


def derive_seed(*parts: object) -> int:
    """A deterministic 31-bit seed computed from arbitrary parts.

    Unlike ``hash()``, the value is the same in every Python process, so a run can be
    reproduced exactly: ``derive_seed(seed, "noise", mechanism, epsilon)``.
    """
    text = "|".join(repr(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") & 0x7FFFFFFF
