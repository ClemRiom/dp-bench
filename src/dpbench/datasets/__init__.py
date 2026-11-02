"""Dataset plugins. Importing this package registers the built-in datasets."""

from dpbench.datasets import adult, cifar10, digits, mnist, synthetic  # noqa: F401
from dpbench.datasets.base import DatasetLoader, cache_dir, download

__all__ = ["DatasetLoader", "cache_dir", "download"]
