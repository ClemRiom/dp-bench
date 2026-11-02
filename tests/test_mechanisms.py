import math

import numpy as np
import pytest

from dpbench.accounting.base import UnsupportedNoiseError
from dpbench.core.registry import MECHANISMS
from dpbench.core.seeding import seed_everything
from dpbench.mechanisms.base import SamplingPlan
from dpbench.models import MLP


def fit(name, dataset, epsilon=4.0, epochs=3, seed=0, global_accountant=None, **params):
    seed_everything(seed)
    model = MLP(dataset.input_shape, dataset.n_classes, hidden=[16])
    mechanism = MECHANISMS.get(name)(**params)
    result = mechanism.fit(
        model,
        dataset.X_train,
        dataset.y_train,
        target_epsilon=epsilon,
        delta=1e-5,
        epochs=epochs,
        batch_size=32,
        seed=seed,
        accountant=global_accountant,
    )
    accuracy = float(np.mean(mechanism.predict(dataset.X_test) == dataset.y_test))
    return mechanism, result, accuracy


def test_sampling_plan_matches_opacus():
    plan = SamplingPlan(n=1000, batch_size=64, epochs=3)
    assert plan.steps_per_epoch == 16
    assert plan.sample_rate == pytest.approx(1 / 16)
    assert plan.steps == 48
    assert plan.expected_batch_size == 62


def test_non_private_baseline_learns(tiny_dataset):
    _, result, accuracy = fit("non_private", tiny_dataset, lr=0.5)
    assert accuracy > 0.85
    assert math.isinf(result.epsilon)
    assert len(result.history) == 3


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, params",
    [
        ("dpsgd", {"lr": 0.5}),
        ("dp_adam", {"lr": 0.01}),
        ("dp_lap_sgd", {"lr": 0.5, "max_grad_norm": 3.0}),
        ("dpsgd_opacus", {"lr": 0.5}),
    ],
)
def test_private_mechanisms_stay_within_budget(tiny_dataset, name, params):
    mechanism, result, accuracy = fit(name, tiny_dataset, **params)
    assert result.epsilon <= 4.0
    assert result.epsilon > 3.9  # the calibrated noise is not wastefully large
    assert result.noise_multiplier > 0
    assert mechanism.get_privacy_spent() == (result.epsilon, 1e-5)
    assert accuracy > 0.6
    assert len(result.history) == 3
    assert math.isclose(result.history[-1]["epsilon"], result.epsilon, rel_tol=1e-9)


def test_dp_lap_sgd_uses_a_laplace_accountant(tiny_dataset):
    _, result, _ = fit("dp_lap_sgd", tiny_dataset, epochs=1, global_accountant="rdp")  # RDP has no Laplace: fallback
    assert result.accountant == "pld"
    with pytest.raises(UnsupportedNoiseError):
        fit("dp_lap_sgd", tiny_dataset, epochs=1, accountant="rdp")  # an explicit choice is an error


def test_global_accountant_is_used_when_compatible(tiny_dataset):
    _, result, _ = fit("dpsgd", tiny_dataset, epochs=1, global_accountant="rdp")
    assert result.accountant == "rdp"


def test_training_is_reproducible(tiny_dataset):
    first, _, _ = fit("dpsgd", tiny_dataset, epochs=2, seed=3)
    second, _, _ = fit("dpsgd", tiny_dataset, epochs=2, seed=3)
    np.testing.assert_allclose(first.predict_proba(tiny_dataset.X_test), second.predict_proba(tiny_dataset.X_test))
    third, _, _ = fit("dpsgd", tiny_dataset, epochs=2, seed=4)
    assert not np.allclose(first.predict_proba(tiny_dataset.X_test), third.predict_proba(tiny_dataset.X_test))


@pytest.mark.slow
def test_in_house_engine_and_opacus_agree(tiny_dataset):
    _, ours, accuracy_ours = fit("dpsgd", tiny_dataset, epsilon=8.0, epochs=4)
    _, theirs, accuracy_theirs = fit("dpsgd_opacus", tiny_dataset, epsilon=8.0, epochs=4)
    assert ours.noise_multiplier == theirs.noise_multiplier
    assert ours.epsilon == pytest.approx(theirs.extra["epsilon_opacus"], rel=0.02)
    assert abs(accuracy_ours - accuracy_theirs) < 0.08


def test_fixed_noise_multiplier_skips_calibration(tiny_dataset):
    _, result, _ = fit("dpsgd", tiny_dataset, epochs=1, noise_multiplier=2.0)
    assert result.noise_multiplier == 2.0
    assert result.epsilon > 0


def test_predict_before_fit_fails():
    with pytest.raises(RuntimeError):
        MECHANISMS.get("dpsgd")().predict_proba(np.zeros((1, 3), dtype=np.float32))
