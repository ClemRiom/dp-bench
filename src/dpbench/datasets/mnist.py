"""MNIST and Fashion-MNIST (IDX files, 28x28 grey images, 10 classes)."""

from __future__ import annotations

import gzip
import math
from pathlib import Path
from typing import ClassVar

import numpy as np

from dpbench.core.registry import register_dataset
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader, cache_dir, download

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def read_idx(path: Path) -> np.ndarray:
    """Read a gzip-compressed IDX file of unsigned bytes (the MNIST format)."""
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 4:
        raise ValueError(f"{path} is not an IDX file")
    zero, dtype_code, ndim = int.from_bytes(data[:2], "big"), data[2], data[3]
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path} is not an IDX file of unsigned bytes")
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    offset = 4 + 4 * ndim
    if len(data) - offset != math.prod(shape):
        raise ValueError(f"{path}: the header announces shape {shape} but the file has another size")
    return np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(shape)


class IDXImageDataset(DatasetLoader):
    """Base class of the datasets distributed as four IDX files (MNIST and its clones)."""

    MIRRORS: ClassVar[tuple[str, ...]] = ()
    MD5: ClassVar[dict[str, str]] = {}
    MEAN: ClassVar[float] = 0.5
    STD: ClassVar[float] = 0.5
    CLASS_NAMES: ClassVar[list[str]] = [str(k) for k in range(10)]
    DESCRIPTION: ClassVar[str] = ""
    SOURCE: ClassVar[str] = ""

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
        images = array.astype(np.float32)[:, None, :, :] / 255.0
        if self.normalize:  # published constants, not statistics of the private data
            images = (images - self.MEAN) / self.STD
        return images

    def load_full(self) -> Dataset:
        directory = cache_dir(self.name)
        arrays = {
            key: read_idx(download(filename, self.MIRRORS, directory, md5=self.MD5.get(filename)))
            for key, filename in FILES.items()
        }
        return Dataset(
            name=self.name,
            X_train=self._images(arrays["train_images"]),
            y_train=arrays["train_labels"].astype(np.int64),
            X_test=self._images(arrays["test_images"]),
            y_test=arrays["test_labels"].astype(np.int64),
            n_classes=10,
            class_names=list(self.CLASS_NAMES),
            description=self.DESCRIPTION,
            source=self.SOURCE,
        )


@register_dataset("mnist")
class MNIST(IDXImageDataset):
    """MNIST handwritten digits: 60,000 training and 10,000 test grey images of 28x28 pixels, 10 classes."""

    MIRRORS = (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://raw.githubusercontent.com/fgnt/mnist/master/",
    )
    MD5 = {
        "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    }
    MEAN, STD = 0.1307, 0.3081
    DESCRIPTION = "MNIST handwritten digits, 28x28 grey images"
    SOURCE = "LeCun, Cortes & Burges (1998)"


@register_dataset("fashion_mnist")
class FashionMNIST(IDXImageDataset):
    """Fashion-MNIST (Zalando): 60,000 training and 10,000 test grey images of clothes, 28x28 pixels, 10 classes."""

    MIRRORS = (
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    )
    MD5 = {
        "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
        "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
        "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
    }
    MEAN, STD = 0.2860, 0.3530
    CLASS_NAMES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    DESCRIPTION = "Fashion-MNIST clothes, 28x28 grey images"
    SOURCE = "Xiao, Rasul & Vollgraf (2017)"
