"""GeoDP: hyperspherical coordinates, noise scales, and the two privacy accountings."""

import math

import numpy as np
import pytest
import torch

from dpbench.core.registry import MECHANISMS
from dpbench.core.seeding import seed_everything
from dpbench.mechanisms.base import SamplingPlan
from dpbench.mechanisms.geodp import from_spherical, noise_scales, to_spherical
from dpbench.models import MLP


@pytest.mark.parametrize("d", [2, 3, 10, 1000])
def test_spherical_round_trip(d):
    g = torch.randn(d, dtype=torch.float64, generator=torch.Generator().manual_seed(d))
    g[-1] = -abs(g[-1])  # the last angle keeps the sign of the last coordinate
    radius, theta = to_spherical(g)
    assert radius == pytest.approx(float(g.norm()))
    assert theta.numel() == d - 1
    assert torch.all(theta[:-1] >= 0) and torch.all(theta[:-1] <= math.pi)
    assert torch.allclose(from_spherical(radius, theta), g, atol=1e-10)


def test_spherical_coordinates_in_three_dimensions():
    g = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    radius, theta = to_spherical(g)
    assert float(radius) == 2.0
    assert theta.tolist() == pytest.approx([math.pi / 2, math.pi / 2])  # the paper's example (Fig. 2)


def test_noise_scales_of_the_paper():
    radial, angular = noise_scales(max_grad_norm=1.0, sigma=2.0, beta=0.1, dimension=98, batch_size=50)
    assert radial == pytest.approx(1.0 / 50 * 2.0)  # (C / B) σ
    assert angular == pytest.approx(math.sqrt(100) * 0.1 * math.pi / 50 * 2.0)  # (√(d+2) β π / B) σ


def test_worst_case_factor():
    geodp = MECHANISMS.get("geodp")(beta=0.1)
    plan = SamplingPlan(n=1000, batch_size=50, epochs=1)
    assert geodp.worst_case_factor(plan) == pytest.approx(math.sqrt(1 + (plan.expected_batch_size / 0.1) ** 2))
    with pytest.raises(ValueError):
        MECHANISMS.get("geodp")(accounting="optimistic")
    with pytest.raises(ValueError):
        MECHANISMS.get("geodp")(beta=0)


def fit(dataset, **params):
    seed_everything(0)
    model = MLP(dataset.input_shape, dataset.n_classes, hidden=[8])
    mechanism = MECHANISMS.get("geodp")(lr=0.5, **params)
    result = mechanism.fit(
        model, dataset.X_train, dataset.y_train, target_epsilon=2.0, delta=1e-5, epochs=2, batch_size=32, seed=0
    )
    return mechanism, result


def test_geodp_reports_the_claimed_and_the_certified_epsilon(tiny_dataset):
    _, paper = fit(tiny_dataset, beta=0.1, accounting="paper")
    assert paper.epsilon <= 2.0 + 1e-9
    assert paper.extra["epsilon_paper_claim"] == pytest.approx(paper.epsilon)
    assert paper.extra["epsilon_worst_case"] > 100 * paper.epsilon  # the claim does not hold in the worst case
    assert paper.extra["noise_std_radius"] > 0 and paper.extra["noise_std_angle"] > 0

    _, worst = fit(tiny_dataset, beta=0.1, accounting="worst_case")
    assert worst.extra["epsilon_worst_case"] == pytest.approx(worst.epsilon)
    assert worst.epsilon <= 2.0 + 1e-9
    # Certified: the noise is √(1 + (B/β)²) times larger than with the paper's accounting.
    plan = SamplingPlan(len(tiny_dataset.y_train), 32, 2)
    ratio = worst.noise_multiplier / paper.noise_multiplier
    assert ratio == pytest.approx(math.sqrt(1 + (plan.expected_batch_size / 0.1) ** 2), rel=1e-6)


def test_geodp_without_noise_follows_the_clipped_gradient(tiny_dataset):
    from dpbench.mechanisms.geodp import GeometricPrivateGradient

    model = MLP(tiny_dataset.input_shape, tiny_dataset.n_classes, hidden=[4])
    engine = GeometricPrivateGradient(model, beta=0.5, max_grad_norm=1.0, noise_multiplier=0.0)
    g = torch.randn(50, dtype=torch.float64)
    assert torch.allclose(engine.perturb(g, 32), g, atol=1e-10)
    engine.noise_multiplier = 1.0
    assert not torch.allclose(engine.perturb(g, 32), g)


def test_geodp_learns(tiny_dataset):
    mechanism, _ = fit(tiny_dataset, beta=0.1, accounting="paper")
    accuracy = float(np.mean(mechanism.predict(tiny_dataset.X_test) == tiny_dataset.y_test))
    assert accuracy > 0.6
