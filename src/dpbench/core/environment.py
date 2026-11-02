"""Environment snapshot saved with every run (package versions, hardware, git commit)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "torch",
    "opacus",
    "dp-accounting",
    "omegaconf",
    "plotly",
    "streamlit",
)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def git_info() -> dict[str, Any] | None:
    """Commit and dirty flag of the DP-Bench checkout, if it is a git repository."""
    here = Path(__file__).resolve().parent
    try:
        top = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        if not (Path(top) / "src" / "dpbench").is_dir():
            return None
        commit = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", top, "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": bool(dirty)}


def environment_snapshot() -> dict[str, Any]:
    import torch

    from dpbench import __version__

    cuda = torch.cuda.is_available()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dpbench": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "packages": {name: _version(name) for name in PACKAGES},
        "git": git_info(),
    }
