import math

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from dpbench.core.registry import METRICS


def metric(name):
    return METRICS.get(name)()


def test_utility_metrics_match_scikit_learn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    logits = rng.normal(size=(200, 3)) + 2 * np.eye(3)[y]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert metric("accuracy")(y, proba) == pytest.approx(np.mean(proba.argmax(1) == y))
    assert metric("f1_macro")(y, proba) == pytest.approx(f1_score(y, proba.argmax(1), average="macro"))
    assert metric("roc_auc")(y, proba) == pytest.approx(roc_auc_score(y, proba, multi_class="ovr"))
    assert metric("loss")(y, proba) == pytest.approx(-np.mean(np.log(proba[np.arange(200), y])))


def test_fairness_metrics():
    y = np.array([0, 1, 0, 1, 0, 1])
    proba = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1]], dtype=float)
    groups = np.array([0, 0, 0, 1, 1, 1])
    assert metric("fairness_gap")(y, proba, groups) == pytest.approx(1 - 2 / 3)
    assert metric("demographic_parity_gap")(y, proba, groups) == pytest.approx(0.0)
    assert metric("class_accuracy_gap")(y, proba) == pytest.approx(1.0 - 2 / 3)
    assert math.isnan(metric("fairness_gap")(y, proba, None))
