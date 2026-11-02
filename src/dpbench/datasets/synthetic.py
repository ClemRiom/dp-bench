"""Synthetic tabular data generated on the fly (no download)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from dpbench.core.registry import register_dataset
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader


@register_dataset("synthetic")
class Synthetic(DatasetLoader):
    """Synthetic tabular classification task generated on the fly (offline; used by the tests).

    Gaussian clusters from scikit-learn's ``make_classification``. A minority group
    (``minority_fraction`` of the records) has shifted features, so that the fairness
    metrics have a gap to measure.
    """

    def __init__(
        self,
        n_samples: int = 3000,
        n_features: int = 20,
        n_classes: int = 2,
        class_sep: float = 1.0,
        minority_fraction: float = 0.3,
        minority_shift: float = 1.0,
        test_size: float = 0.25,
        seed: int = 0,
        n_train: int | None = None,
        n_test: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        super().__init__(n_train, n_test, subset_seed)
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_sep = class_sep
        self.minority_fraction = minority_fraction
        self.minority_shift = minority_shift
        self.test_size = test_size
        self.seed = seed

    def load_full(self) -> Dataset:
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=max(2, self.n_features // 2),
            n_redundant=0,
            n_classes=self.n_classes,
            n_clusters_per_class=1,
            class_sep=self.class_sep,
            flip_y=0.01,
            random_state=self.seed,
        )
        rng = np.random.default_rng(self.seed)
        groups = (rng.random(self.n_samples) < self.minority_fraction).astype(np.int64)
        X[groups == 1] += self.minority_shift * rng.normal(size=self.n_features)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X, y, groups, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        return Dataset(
            name="synthetic",
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            n_classes=self.n_classes,
            class_names=[f"class {k}" for k in range(self.n_classes)],
            feature_names=[f"x{i}" for i in range(self.n_features)],
            sensitive_attribute="group",
            groups_train=g_tr,
            groups_test=g_te,
            group_names=["majority", "minority"],
            description=f"Synthetic Gaussian clusters ({self.n_samples} records, {self.n_features} features)",
            source="scikit-learn make_classification",
        )
