"""The private gradient of DP-SGD: per-example gradients, clipping and noise.

For a batch ``B`` drawn with Poisson sampling (each of the ``N`` training examples is kept
independently with probability ``q``), DP-SGD replaces the gradient by::

    g = ( sum_{i in B} clip_C( grad loss(theta; x_i, y_i) ) + noise ) / (q N)

``clip_C`` rescales every per-example gradient whose norm exceeds ``C`` to norm ``C`` (L2
norm with Gaussian noise, L1 norm with Laplace noise). Adding or removing one example
therefore changes the sum by a vector of norm at most ``C``: the sensitivity of the sum is
``C``. The noise has standard deviation ``sigma * C`` per coordinate (Gaussian) or scale
``sigma * C`` (Laplace), where ``sigma`` is the noise multiplier chosen by the accountant.
The sum is divided by the *expected* batch size ``q N`` and not by ``|B|``, because ``|B|``
itself depends on the private data.

Per-example gradients are computed in one vectorised pass with ``torch.func``
(``vmap(grad(loss of one example))``), in chunks of ``max_physical_batch_size`` examples to
bound memory.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

NOISES = ("gaussian", "laplace")


class PrivateGradient:
    """Computes the DP-SGD gradient of a batch and stores it in the ``.grad`` of the parameters."""

    def __init__(
        self,
        model: nn.Module,
        *,
        max_grad_norm: float,
        noise_multiplier: float,
        noise: str = "gaussian",
        norm: int = 2,
        max_physical_batch_size: int = 512,
        generator: torch.Generator | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy,
    ) -> None:
        if noise not in NOISES:
            raise ValueError(f"noise must be one of {NOISES}, got {noise!r}")
        if norm not in (1, 2):
            raise ValueError(f"norm must be 1 or 2, got {norm}")
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be non-negative, got {noise_multiplier}")
        self.model = model
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.noise = noise
        self.norm = norm
        self.max_physical_batch_size = int(max_physical_batch_size)
        self.generator = generator

        def loss_of_one_example(params, buffers, x, y):
            output = functional_call(model, (params, buffers), (x.unsqueeze(0),))
            return loss_fn(output, y.unsqueeze(0))

        self._per_example = vmap(grad_and_value(loss_of_one_example), in_dims=(None, None, 0, 0))

    def _parameters(self) -> dict[str, torch.Tensor]:
        return {name: p for name, p in self.model.named_parameters() if p.requires_grad}

    def per_example_gradients(self, x: torch.Tensor, y: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Gradients of every example's loss (leading batch dimension) and the losses."""
        params = {name: p.detach() for name, p in self._parameters().items()}
        buffers = {name: b.detach() for name, b in self.model.named_buffers()}
        return self._per_example(params, buffers, x, y)

    def norms(self, grads: dict[str, torch.Tensor]) -> torch.Tensor:
        """L1 or L2 norm of each example's gradient, all parameters concatenated."""
        flat = [g.reshape(g.shape[0], -1) for g in grads.values()]
        if self.norm == 2:
            return torch.sqrt(sum((f * f).sum(dim=1) for f in flat))
        return sum(f.abs().sum(dim=1) for f in flat)

    def clip_factors(self, norms: torch.Tensor) -> torch.Tensor:
        return (self.max_grad_norm / (norms + 1e-12)).clamp(max=1.0)

    def clipped_sum(self, x: torch.Tensor, y: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Sum of the clipped per-example gradients of the batch, with statistics."""
        total = {name: torch.zeros_like(p) for name, p in self._parameters().items()}
        clipped, norm_sum, loss_sum = 0, 0.0, 0.0
        for start in range(0, len(x), self.max_physical_batch_size):
            stop = start + self.max_physical_batch_size
            grads, losses = self.per_example_gradients(x[start:stop], y[start:stop])
            norms = self.norms(grads)
            factors = self.clip_factors(norms)
            for name, g in grads.items():
                total[name] += torch.tensordot(factors, g, dims=1)
            clipped += int((norms > self.max_grad_norm).sum())
            norm_sum += float(norms.sum())
            loss_sum += float(losses.sum())
        stats = {"examples": float(len(x)), "clipped": float(clipped), "norm_sum": norm_sum, "loss_sum": loss_sum}
        return total, stats

    def sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        """Noise with the shape of ``like``: N(0, (sigma C)^2) or Laplace(0, sigma C) per coordinate."""
        scale = self.noise_multiplier * self.max_grad_norm
        if scale == 0:
            return torch.zeros_like(like)
        if self.noise == "gaussian":
            return scale * torch.randn(like.shape, generator=self.generator, device=like.device, dtype=like.dtype)
        first = torch.empty(like.shape, device=like.device, dtype=like.dtype).exponential_(generator=self.generator)
        second = torch.empty(like.shape, device=like.device, dtype=like.dtype).exponential_(generator=self.generator)
        return scale * (first - second)

    def privatize(self, x: torch.Tensor, y: torch.Tensor, expected_batch_size: float) -> dict[str, float]:
        """Store ``(clipped sum + noise) / expected_batch_size`` in ``p.grad`` for every parameter."""
        total, stats = self.clipped_sum(x, y)
        for name, p in self._parameters().items():
            p.grad = (total[name] + self.sample_noise(total[name])) / expected_batch_size
        return stats
