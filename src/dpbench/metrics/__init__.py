"""Metric plugins: utility (accuracy, F1, ROC AUC, loss) and fairness (gaps between groups).

A metric is called as ``metric(y_true, proba, groups)`` where ``proba`` holds the predicted
class probabilities and ``groups`` the sensitive attribute of each example (or ``None``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from dpbench.core.registry import register_metric


def _normalise(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    return proba / proba.sum(axis=1, keepdims=True)


class Metric(ABC):
    """Base class of the metric plugins."""

    name: ClassVar[str]
    label: ClassVar[str] = ""
    higher_is_better: ClassVar[bool] = True
    requires_groups: ClassVar[bool] = False
    #: Comparison axis the metric belongs to ("utility" or "fairness").
    axis: ClassVar[str] = "utility"

    @abstractmethod
    def __call__(self, y_true: np.ndarray, proba: np.ndarray, groups: np.ndarray | None = None) -> float:
        """Value of the metric on the given predictions."""


@register_metric("accuracy")
class Accuracy(Metric):
    """Fraction of examples whose most probable class is the true class."""

    label = "Accuracy"

    def __call__(self, y_true, proba, groups=None):
        return float(np.mean(np.argmax(proba, axis=1) == y_true))


@register_metric("f1_macro")
class F1Macro(Metric):
    """F1 score averaged over the classes with equal weights (robust to class imbalance)."""

    label = "Macro F1"

    def __call__(self, y_true, proba, groups=None):
        labels = np.arange(np.asarray(proba).shape[1])
        return float(f1_score(y_true, np.argmax(proba, axis=1), labels=labels, average="macro", zero_division=0))


@register_metric("roc_auc")
class ROCAUC(Metric):
    """Area under the ROC curve (one-vs-rest macro average for more than two classes)."""

    label = "ROC AUC"

    def __call__(self, y_true, proba, groups=None):
        proba = _normalise(proba)
        try:
            if proba.shape[1] == 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=np.arange(proba.shape[1]))
            )
        except ValueError:  # a class is missing from y_true
            return math.nan


@register_metric("loss")
class CrossEntropy(Metric):
    """Mean cross-entropy (negative log-likelihood of the true class)."""

    label = "Cross-entropy"
    higher_is_better = False

    def __call__(self, y_true, proba, groups=None):
        proba = _normalise(proba)
        return float(-np.mean(np.log(np.clip(proba[np.arange(len(y_true)), y_true], 1e-15, None))))


def _per_group(values: np.ndarray, groups: np.ndarray) -> list[float]:
    return [float(values[groups == g].mean()) for g in np.unique(groups)]


@register_metric("fairness_gap")
class AccuracyGap(Metric):
    """Accuracy gap between the best and the worst group of the sensitive attribute (0 = equal accuracy)."""

    label = "Accuracy gap between groups"
    higher_is_better = False
    requires_groups = True
    axis = "fairness"

    def __call__(self, y_true, proba, groups=None):
        if groups is None:
            return math.nan
        accuracies = _per_group(np.argmax(proba, axis=1) == y_true, groups)
        return max(accuracies) - min(accuracies)


@register_metric("demographic_parity_gap")
class DemographicParityGap(Metric):
    """Gap of positive prediction rate between groups (binary tasks only; 0 = demographic parity)."""

    label = "Demographic parity gap"
    higher_is_better = False
    requires_groups = True
    axis = "fairness"

    def __call__(self, y_true, proba, groups=None):
        if groups is None or np.asarray(proba).shape[1] != 2:
            return math.nan
        rates = _per_group(np.argmax(proba, axis=1) == 1, groups)
        return max(rates) - min(rates)


@register_metric("class_accuracy_gap")
class ClassAccuracyGap(Metric):
    """Gap between the best and the worst per-class accuracy (DP tends to hurt hard or rare classes most)."""

    label = "Accuracy gap between classes"
    higher_is_better = False
    axis = "fairness"

    def __call__(self, y_true, proba, groups=None):
        recalls = _per_group(np.argmax(proba, axis=1) == y_true, np.asarray(y_true))
        return max(recalls) - min(recalls)


__all__ = ["Metric"]
