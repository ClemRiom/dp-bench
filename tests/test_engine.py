import math

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from dpbench.core.seeding import torch_generator
from dpbench.mechanisms.engine import PrivateGradient
from dpbench.models import CNN, MLP


def mlp():
    torch.manual_seed(0)
    return MLP((6,), 3, hidden=[5])


def data(n, scale=1.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = scale * torch.randn(n, 6, generator=generator)
    y = torch.randint(0, 3, (n,), generator=generator)
    return x, y


def loop_gradients(model, x, y):
    grads = []
    for i in range(len(x)):
        model.zero_grad()
        loss = F.cross_entropy(model(x[i : i + 1]), y[i : i + 1])
        loss.backward()
        grads.append({name: p.grad.detach().clone() for name, p in model.named_parameters()})
    return grads


@pytest.mark.parametrize("make_model, shape", [(mlp, (6,)), (lambda: CNN((1, 28, 28), 10), (1, 28, 28))])
def test_per_example_gradients_equal_a_python_loop(make_model, shape):
    model = make_model()
    torch.manual_seed(1)
    x = torch.randn(5, *shape)
    y = torch.randint(0, 3, (5,))
    engine = PrivateGradient(model, max_grad_norm=1.0, noise_multiplier=0.0)
    grads, losses = engine.per_example_gradients(x, y)
    expected = loop_gradients(model, x, y)
    for i in range(len(x)):
        for name, g in grads.items():
            torch.testing.assert_close(g[i], expected[i][name], rtol=1e-4, atol=1e-6)
        assert losses[i].item() == pytest.approx(F.cross_entropy(model(x[i : i + 1]), y[i : i + 1]).item(), rel=1e-5)


@settings(max_examples=40, deadline=None)
@given(
    max_norm=st.floats(0.01, 10.0),
    scale=st.floats(0.1, 100.0),
    batch=st.integers(1, 9),
    norm=st.sampled_from([1, 2]),
    seed=st.integers(0, 1000),
)
def test_clipping_bounds_every_example(max_norm, scale, batch, norm, seed):
    model = mlp()
    x, y = data(batch, scale, seed)
    engine = PrivateGradient(model, max_grad_norm=max_norm, noise_multiplier=0.0, norm=norm)
    grads, _ = engine.per_example_gradients(x, y)
    factors = engine.clip_factors(engine.norms(grads))
    clipped = {name: g * factors.view(-1, *[1] * (g.dim() - 1)) for name, g in grads.items()}
    assert bool((engine.norms(clipped) <= max_norm * (1 + 1e-5)).all())
    total, stats = engine.clipped_sum(x, y)
    for name in total:
        torch.testing.assert_close(total[name], clipped[name].sum(0), rtol=1e-4, atol=1e-6)
    assert stats["examples"] == batch


@pytest.mark.parametrize("norm", [1, 2])
def test_one_example_changes_the_clipped_sum_by_at_most_the_bound(norm):
    model = mlp()
    x, y = data(9, scale=30.0)
    engine = PrivateGradient(model, max_grad_norm=0.7, noise_multiplier=0.0, norm=norm)
    with_all, _ = engine.clipped_sum(x, y)
    without_last, _ = engine.clipped_sum(x[:-1], y[:-1])
    difference = {name: (with_all[name] - without_last[name]).unsqueeze(0) for name in with_all}
    assert engine.norms(difference).item() <= 0.7 * (1 + 1e-5)


def test_without_noise_and_clipping_the_step_is_plain_gradient_descent():
    model = mlp()
    x, y = data(8)
    engine = PrivateGradient(model, max_grad_norm=1e9, noise_multiplier=0.0, max_physical_batch_size=3)
    engine.privatize(x, y, expected_batch_size=len(x))
    private = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    for name, p in model.named_parameters():
        torch.testing.assert_close(private[name], p.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("noise, expected_std", [("gaussian", 3.0), ("laplace", 3.0 * math.sqrt(2))])
def test_noise_has_the_calibrated_scale(noise, expected_std):
    engine = PrivateGradient(
        mlp(),
        max_grad_norm=2.0,
        noise_multiplier=1.5,
        noise=noise,
        norm=2 if noise == "gaussian" else 1,
        generator=torch_generator(0),
    )
    sample = engine.sample_noise(torch.zeros(400_000))
    assert sample.mean().item() == pytest.approx(0.0, abs=0.02)
    assert sample.std().item() == pytest.approx(expected_std, rel=0.01)
    if noise == "laplace":
        assert sample.abs().mean().item() == pytest.approx(3.0, rel=0.01)


def test_noise_is_reproducible():
    first = PrivateGradient(mlp(), max_grad_norm=1.0, noise_multiplier=1.0, generator=torch_generator(7))
    second = PrivateGradient(mlp(), max_grad_norm=1.0, noise_multiplier=1.0, generator=torch_generator(7))
    torch.testing.assert_close(first.sample_noise(torch.zeros(100)), second.sample_noise(torch.zeros(100)))


def test_an_empty_batch_releases_pure_noise():
    model = mlp()
    engine = PrivateGradient(model, max_grad_norm=1.0, noise_multiplier=1.0, generator=torch_generator(0))
    x, y = data(0)
    stats = engine.privatize(x, y, expected_batch_size=10)
    assert stats["examples"] == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_arguments():
    with pytest.raises(ValueError):
        PrivateGradient(mlp(), max_grad_norm=0.0, noise_multiplier=1.0)
    with pytest.raises(ValueError):
        PrivateGradient(mlp(), max_grad_norm=1.0, noise_multiplier=1.0, noise="uniform")
