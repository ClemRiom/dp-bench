"""Shared helpers of the dataset loaders: cache directory, downloads with mirrors and MD5 checks."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np

from dpbench.core.types import Dataset

logger = logging.getLogger(__name__)

USER_AGENT = "dp-bench/0.1"


class DownloadError(RuntimeError):
    """No mirror could provide a dataset file."""


def cache_dir(name: str | None = None) -> Path:
    """``$DPBENCH_DATA`` (default ``~/.cache/dpbench``), or one of its sub-directories."""
    root = Path(os.environ.get("DPBENCH_DATA", "~/.cache/dpbench")).expanduser()
    path = root / name if name else root
    path.mkdir(parents=True, exist_ok=True)
    return path


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    filename: str,
    mirrors: Sequence[str],
    directory: Path,
    md5: str | None = None,
    timeout: float = 60.0,
) -> Path:
    """Return ``directory / filename``, downloading it from the first mirror that works.

    A mirror is a base URL (the file name is appended) or a full URL ending with the file
    name. A cached file is reused when its MD5 matches; a downloaded file with the wrong MD5
    is discarded and the next mirror is tried.
    """
    target = directory / filename
    if target.is_file() and (md5 is None or md5sum(target) == md5):
        return target
    errors = []
    for mirror in mirrors:
        url = mirror if mirror.endswith("/" + filename) else mirror.rstrip("/") + "/" + filename
        partial = target.with_name(target.name + ".part")
        try:
            logger.info("Downloading %s", url)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out)
            if md5 is not None:
                actual = md5sum(partial)
                if actual != md5:
                    raise DownloadError(f"MD5 mismatch: got {actual}, expected {md5}")
            partial.replace(target)
            return target
        except Exception as error:  # try the next mirror
            partial.unlink(missing_ok=True)
            errors.append(f"{url}: {error}")
            logger.warning("Download failed: %s: %s", url, error)
    raise DownloadError(
        f"Could not download {filename}. Tried:\n  "
        + "\n  ".join(errors)
        + f"\nDownload the file manually into {directory} (or set DPBENCH_DATA to another cache directory)."
    )


def subsample(dataset: Dataset, n_train: int | None, n_test: int | None, seed: int) -> Dataset:
    """Keep a uniformly random subset of the training and/or test examples."""
    rng = np.random.default_rng(seed)
    changes = {}
    for split, n in (("train", n_train), ("test", n_test)):
        X = getattr(dataset, f"X_{split}")
        if n is None or n >= len(X):
            continue
        if n < 1:
            raise ValueError(f"n_{split} must be positive, got {n}")
        index = np.sort(rng.choice(len(X), size=n, replace=False))
        changes[f"X_{split}"] = X[index]
        changes[f"y_{split}"] = getattr(dataset, f"y_{split}")[index]
        groups = getattr(dataset, f"groups_{split}")
        if groups is not None:
            changes[f"groups_{split}"] = groups[index]
    return replace(dataset, **changes) if changes else dataset


class DatasetLoader(ABC):
    """Base class of the dataset plugins.

    A subclass implements :meth:`load_full`, which returns the whole dataset already split
    into training and test sets. ``n_train`` and ``n_test`` keep a random subset of each
    split (handy for quick experiments); ``subset_seed`` fixes that subset.
    """

    name: ClassVar[str]

    def __init__(self, n_train: int | None = None, n_test: int | None = None, subset_seed: int = 0) -> None:
        self.n_train = n_train
        self.n_test = n_test
        self.subset_seed = subset_seed

    @abstractmethod
    def load_full(self) -> Dataset:
        """The complete dataset."""

    def load(self) -> Dataset:
        return subsample(self.load_full(), self.n_train, self.n_test, self.subset_seed)
