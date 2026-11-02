"""Data structures shared by the plugins."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """A classification dataset split into a training set and a test set.

    Features are float32 arrays of shape ``(n, d)`` for tabular data and
    ``(n, channels, height, width)`` for images; labels are int64 class indices in
    ``[0, n_classes)``. ``groups_train`` and ``groups_test`` hold the integer code of a
    sensitive attribute (for example sex in Adult), used by the fairness metrics.
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    class_names: list[str] | None = None
    feature_names: list[str] | None = None
    sensitive_attribute: str | None = None
    groups_train: np.ndarray | None = None
    groups_test: np.ndarray | None = None
    group_names: list[str] | None = None
    description: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        self.X_train = np.ascontiguousarray(self.X_train, dtype=np.float32)
        self.X_test = np.ascontiguousarray(self.X_test, dtype=np.float32)
        self.y_train = np.asarray(self.y_train, dtype=np.int64)
        self.y_test = np.asarray(self.y_test, dtype=np.int64)
        if len(self.X_train) != len(self.y_train) or len(self.X_test) != len(self.y_test):
            raise ValueError(f"{self.name}: features and labels have different lengths")
        if self.X_train.shape[1:] != self.X_test.shape[1:]:
            raise ValueError(f"{self.name}: training and test features have different shapes")
        for y in (self.y_train, self.y_test):
            if len(y) and (y.min() < 0 or y.max() >= self.n_classes):
                raise ValueError(f"{self.name}: labels must lie in [0, {self.n_classes})")
        for groups, n in ((self.groups_train, len(self.y_train)), (self.groups_test, len(self.y_test))):
            if groups is not None and len(groups) != n:
                raise ValueError(f"{self.name}: group codes and labels have different lengths")
        if self.groups_train is not None:
            self.groups_train = np.asarray(self.groups_train, dtype=np.int64)
        if self.groups_test is not None:
            self.groups_test = np.asarray(self.groups_test, dtype=np.int64)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.X_train.shape[1:])

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def n_features(self) -> int:
        return int(math.prod(self.input_shape))

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable description (saved as ``dataset.json`` in every run)."""
        counts = np.bincount(self.y_train, minlength=self.n_classes) / max(self.n_train, 1)
        summary: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "input_shape": list(self.input_shape),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "class_balance_train": [round(float(c), 4) for c in counts],
            "majority_class_test_accuracy": float(
                np.max(np.bincount(self.y_test, minlength=self.n_classes)) / max(self.n_test, 1)
            ),
            "sensitive_attribute": self.sensitive_attribute,
            "group_names": self.group_names,
        }
        if self.groups_train is not None:
            g = np.bincount(self.groups_train) / max(self.n_train, 1)
            summary["group_balance_train"] = [round(float(v), 4) for v in g]
        return summary


@dataclass
class FitResult:
    """What a mechanism reports after training.

    ``epsilon`` is the privacy budget actually spent according to the accountant
    (``math.inf`` for non-private training). ``history`` has one dict per epoch.
    """

    epsilon: float
    delta: float
    noise_multiplier: float | None = None
    sample_rate: float | None = None
    steps: int = 0
    epochs: int = 0
    batch_size: int = 0
    accountant: str | None = None
    train_time_s: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
