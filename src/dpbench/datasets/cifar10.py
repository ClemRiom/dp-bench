"""CIFAR-10 (32x32 colour images, 10 classes)."""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from dpbench.core.registry import register_dataset
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader, cache_dir, download

ARCHIVE = "cifar-10-python.tar.gz"
MIRRORS = ("https://www.cs.toronto.edu/~kriz/",)
MD5 = "c58f30108f718f92721af3b95e74349a"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that can only rebuild NumPy arrays: a tampered archive cannot run code."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("_codecs", "encode"),  # bytes in protocol-2 pickles written by Python 3
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden object in a CIFAR-10 batch: {module}.{name}")


def read_batches(archive: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Images ``(n, 3, 32, 32)`` as uint8 and labels of the given batches of the archive."""
    images, labels = [], []
    with tarfile.open(archive, "r:gz") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for name in names:
            if name not in members:
                raise ValueError(f"{archive} has no batch {name!r}")
            handle = tar.extractfile(members[name])
            assert handle is not None
            batch = _NumpyOnlyUnpickler(handle, encoding="latin1").load()
            images.append(np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
            labels.append(np.asarray(batch["labels"], dtype=np.int64))
    return np.concatenate(images), np.concatenate(labels)


@register_dataset("cifar10")
class CIFAR10(DatasetLoader):
    """CIFAR-10: 50,000 training and 10,000 test colour images of 32x32 pixels, 10 classes."""

    def __init__(
        self,
        normalize: bool = True,
        n_train: int | None = None,
        n_test: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        super().__init__(n_train, n_test, subset_seed)
        self.normalize = normalize

    def _images(self, array: np.ndarray) -> np.ndarray:
        images = array.astype(np.float32) / 255.0
        if self.normalize:  # published channel statistics
            images = (images - MEAN[None, :, None, None]) / STD[None, :, None, None]
        return images

    def load_full(self) -> Dataset:
        archive = download(ARCHIVE, MIRRORS, cache_dir("cifar10"), md5=MD5)
        X_tr, y_tr = read_batches(archive, [f"data_batch_{i}" for i in range(1, 6)])
        X_te, y_te = read_batches(archive, ["test_batch"])
        return Dataset(
            name="cifar10",
            X_train=self._images(X_tr),
            y_train=y_tr,
            X_test=self._images(X_te),
            y_test=y_te,
            n_classes=10,
            class_names=list(CLASS_NAMES),
            description="CIFAR-10 colour images, 32x32",
            source="Krizhevsky (2009)",
        )
