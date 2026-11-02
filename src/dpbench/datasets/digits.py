"""Handwritten digits bundled with scikit-learn (no download)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from dpbench.core.registry import register_dataset
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader


@register_dataset("digits")
class Digits(DatasetLoader):
    """Handwritten digits of scikit-learn: 1,797 grey images of 8x8 pixels, 10 classes (bundled, no download)."""

    def __init__(
        self,
        test_size: float = 0.2,
        split_seed: int = 0,
        n_train: int | None = None,
        n_test: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        super().__init__(n_train, n_test, subset_seed)
        self.test_size = test_size
        self.split_seed = split_seed

    def load_full(self) -> Dataset:
        data = load_digits()
        X = (data.images / 16.0).astype(np.float32)[:, None, :, :]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, data.target, test_size=self.test_size, random_state=self.split_seed, stratify=data.target
        )
        return Dataset(
            name="digits",
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            n_classes=10,
            class_names=[str(k) for k in range(10)],
            description="Handwritten digits, 8x8 grey images (UCI optdigits test set)",
            source="scikit-learn load_digits",
        )
