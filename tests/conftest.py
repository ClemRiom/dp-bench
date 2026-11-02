from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest
import torch

from dpbench.core.config import config_from_dict
from dpbench.datasets.synthetic import Synthetic

ROOT = Path(__file__).resolve().parents[1]

torch.set_num_threads(min(2, torch.get_num_threads()))

# The tests never read or write the workspace of the current directory.
os.environ["DPBENCH_WORKSPACE"] = tempfile.mkdtemp(prefix="dpbench-tests-workspace-")


@pytest.fixture(scope="session")
def tiny_dataset():
    """600 synthetic records with 8 features, 2 classes and a minority group."""
    return Synthetic(n_samples=600, n_features=8, seed=0).load()


def tiny_config(**changes) -> dict:
    data = {
        "name": "tiny",
        "description": "Tiny end-to-end experiment used by the tests.",
        "seeds": [0],
        "dataset": {"name": "synthetic", "params": {"n_samples": 400, "n_features": 6}},
        "model": {"name": "mlp", "params": {"hidden": [8]}},
        "training": {"epochs": 2, "batch_size": 32},
        "privacy": {"epsilons": [2.0], "delta": 1e-5},
        "mechanisms": [
            {"name": "non_private", "params": {"lr": 0.5}},
            {"name": "dpsgd", "params": {"lr": 0.5}},
        ],
        "metrics": [
            "accuracy",
            "f1_macro",
            "roc_auc",
            "loss",
            "fairness_gap",
            "demographic_parity_gap",
            "class_accuracy_gap",
        ],
    }
    data.update(changes)
    return data


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """A finished run directory of the tiny experiment (shared by the reporting and dashboard tests)."""
    from dpbench.experiments.runner import run_experiment

    root = tmp_path_factory.mktemp("runs")
    config = config_from_dict(tiny_config(privacy={"epsilons": [1.0, 4.0], "delta": 1e-5}, seeds=[0, 1]))
    result = run_experiment(config, root, console=False)
    return result.directory


def finite(values) -> bool:
    return all(math.isfinite(v) for v in values)
