"""GeoDP (Duan, Hu, Ye & Sun 2025): DP-SGD that perturbs the gradient's magnitude and its direction separately.

Reference: J. Duan, H. Hu, Q. Ye, X. Sun, *Analyzing and Optimizing Perturbation of DP-SGD
Geometrically*, IEEE ICDE 2025, pp. 3439-3452 (full version: arXiv:2504.05618).

**The algorithm (Algorithm 1 of the paper).** At every step, after the per-example
clipping of DP-SGD, the *average* clipped gradient ``g`` of the batch (dimension ``d``,
expected batch size ``B``) is written in hyperspherical coordinates:

* a **radius** (magnitude) ``r = ‖g‖``;
* ``d - 1`` **angles** (the direction): ``θ_z = atan2(‖(g_{z+1}, ..., g_d)‖, g_z)`` for
  ``z ≤ d - 2``, in ``[0, π]``, and ``θ_{d-1} = atan2(g_d, g_{d-1})``, in ``(-π, π]``.

Two independent Gaussian noises are added, a radial one and a directional one::

    r*  = r + (C / B) σ N(0, 1)
    θ*  = θ + (√(d + 2) β π / B) σ N(0, I_{d-1})

and ``(r*, θ*)`` is converted back to a gradient. ``β ∈ (0, 1]`` (the "bounding factor")
scales the direction noise; the paper recommends a small β (0.1 gives its best results).

**Its privacy, and how DP-Bench reports it.** The paper calibrates σ like DP-SGD and
states (Theorem 5) that the noisy gradient is (ε, δ + δ′)-DP, where (ε, δ) is the
guarantee of DP-SGD with the same σ and δ′ ≤ 1 - β (Lemma 2). The radius is fine: one
example moves ``g`` by at most ``C/B``, so ``r`` moves by at most ``C/B`` and the radial
noise has multiplier σ. Three steps of the argument for the angles do not hold in the
worst case:

1. the angle sensitivity ``β π √(d + 2)`` assumes that the neighbouring direction stays
   in a window of width ``βπ`` around the true one (β = 1 is the whole angle space); the
   δ′ ≤ 1 - β meant to cover the other cases is close to 1 for a small β;
2. it is divided by ``B`` like the radius: but when the average gradient is small, one
   example can turn its direction completely, whatever ``B``;
3. the radius and the angles are two releases, each with multiplier σ: even under the
   paper's assumptions, their composition has multiplier σ/√2, not σ.

Without these assumptions the angles have sensitivity ``π √(d + 2)``, so their noise
multiplier is only ``β σ / B``, and the pair (radius, angles) is a Gaussian mechanism of
multiplier ``σ / √(1 + (B/β)²)``: this is the ε that DP-Bench certifies.

DP-Bench therefore offers two accountings, chosen with ``accounting``:

* ``"paper"`` (default): σ calibrated as in the paper; ``epsilon_spent`` is the paper's
  claim (its ε; the δ′ is not counted), and ``epsilon_worst_case`` the ε certified;
* ``"worst_case"``: σ calibrated so that the certified ε meets the target (the noise
  becomes very large).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from dpbench.accounting.base import Accountant
from dpbench.core.registry import ACCOUNTANTS, register_mechanism
from dpbench.mechanisms.base import SamplingPlan
from dpbench.mechanisms.dpsgd import DPSGD
from dpbench.mechanisms.engine import PrivateGradient

ACCOUNTINGS = ("paper", "worst_case")


def to_spherical(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Radius and the d - 1 hyperspherical angles of a vector of dimension d >= 2."""
    tail = torch.sqrt(torch.flip(torch.cumsum(torch.flip(g * g, [0]), 0), [0]))  # tail[k] = ‖g[k:]‖
    theta = torch.atan2(tail[1:], g[:-1])  # θ_k = atan2(‖g[k+1:]‖, g[k]) ∈ [0, π]
    theta[-1] = torch.atan2(g[-1], g[-2])  # the last angle keeps the sign of g[d-1]: (-π, π]
    return tail[0], theta


def noise_scales(
    max_grad_norm: float, sigma: float, beta: float, dimension: int, batch_size: float
) -> tuple[float, float]:
    """Standard deviations of GeoDP's radial noise, (C/B) σ, and angular noise, (√(d+2) β π / B) σ."""
    radial = max_grad_norm / batch_size * sigma
    angular = math.sqrt(dimension + 2) * beta * math.pi / batch_size * sigma
    return radial, angular


def from_spherical(radius: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_spherical`: g_k = r sin θ_0 ... sin θ_{k-1} cos θ_k, and g_{d-1} = r sin θ_0 ... sin θ_{d-2}."""
    sines = torch.cumprod(torch.sin(theta), 0)
    prefix = torch.cat([torch.ones(1, dtype=theta.dtype, device=theta.device), sines])
    g = torch.empty(theta.numel() + 1, dtype=theta.dtype, device=theta.device)
    g[:-1] = radius * prefix[:-1] * torch.cos(theta)
    g[-1] = radius * prefix[-1]
    return g


class GeometricPrivateGradient(PrivateGradient):
    """The private gradient of GeoDP: clipped average, then radial and directional Gaussian noise."""

    def __init__(self, model, *, beta: float = 0.1, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if self.noise != "gaussian" or self.norm != 2:
            raise ValueError("GeoDP uses L2 clipping and Gaussian noise")
        self.beta = float(beta)

    def perturb(self, g: torch.Tensor, expected_batch_size: float) -> torch.Tensor:
        """Add the radial and the directional noise to the average clipped gradient ``g`` (float64)."""
        radial, angular = noise_scales(
            self.max_grad_norm, self.noise_multiplier, self.beta, g.numel(), expected_batch_size
        )

        def gaussian(*shape: int) -> torch.Tensor:
            return torch.randn(shape, generator=self.generator, dtype=g.dtype, device=g.device)

        if g.numel() == 1:  # no direction in dimension 1
            return g + radial * gaussian(1)
        radius, theta = to_spherical(g)
        noisy_radius = radius + radial * gaussian(1)[0]
        noisy_theta = theta + angular * gaussian(theta.numel())
        return from_spherical(noisy_radius, noisy_theta)

    def privatize(self, x: torch.Tensor, y: torch.Tensor, expected_batch_size: float) -> dict[str, float]:
        total, stats = self.clipped_sum(x, y)
        parameters = self._parameters()
        flat = torch.cat([total[name].reshape(-1) for name in parameters]).double() / expected_batch_size
        noisy = self.perturb(flat, expected_batch_size)
        offset = 0
        for p in parameters.values():
            p.grad = noisy[offset : offset + p.numel()].reshape(p.shape).to(p.dtype)
            offset += p.numel()
        return stats


@register_mechanism("geodp")
class GeoDP(DPSGD):
    """GeoDP (Duan et al., ICDE 2025): DP-SGD with a radial noise on the gradient's magnitude and a directional noise on its angles.

    ``beta`` scales the direction noise. ``accounting="paper"`` reports the ε claimed by
    the paper (σ calibrated as for DP-SGD), together with ``epsilon_worst_case``, the ε
    certified without the paper's assumptions on the angles (see the module docstring);
    ``accounting="worst_case"`` calibrates the noise on that certified ε.
    """

    title = "GeoDP"
    engine_class = GeometricPrivateGradient
    #: Allowed values of the string parameters (used by the dashboard's form).
    parameter_choices = {"accounting": ACCOUNTINGS}

    def __init__(self, beta: float = 0.1, accounting: str = "paper", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if accounting not in ACCOUNTINGS:
            raise ValueError(f"accounting must be one of {ACCOUNTINGS}, got {accounting!r}")
        if not 0 < beta <= 1:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        self.beta = float(beta)
        self.accounting = accounting

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {"beta": 0.1, "accounting": "paper", **DPSGD.default_params()}

    def engine_options(self) -> dict[str, Any]:
        return {"beta": self.beta}

    def worst_case_factor(self, plan: SamplingPlan) -> float:
        """σ / (multiplier of the pair (radius, angles) without the 1/B assumption) = √(1 + (B/β)²)."""
        return math.sqrt(1.0 + (plan.expected_batch_size / self.beta) ** 2)

    def accounted_noise_multiplier(self, sigma: float, plan: SamplingPlan) -> float:
        return sigma / self.worst_case_factor(plan) if self.accounting == "worst_case" else sigma

    def noise_multiplier_for(self, accounted: float, plan: SamplingPlan) -> float:
        return accounted * self.worst_case_factor(plan) if self.accounting == "worst_case" else accounted

    def privacy_report(self, accountant: Accountant, sigma: float, plan: SamplingPlan, delta: float) -> dict[str, Any]:
        worst = sigma / self.worst_case_factor(plan)
        report: dict[str, Any] = {
            "geodp_beta": self.beta,
            "geodp_accounting": self.accounting,
            "epsilon_paper_claim": accountant.epsilon(sigma, plan.sample_rate, plan.steps, delta),
            "noise_multiplier_worst_case": worst,
        }
        try:  # tiny multipliers can be out of the range of the numerical accountants: fall back to RDP
            report["epsilon_worst_case"] = accountant.epsilon(worst, plan.sample_rate, plan.steps, delta)
        except Exception:
            try:
                report["epsilon_worst_case"] = ACCOUNTANTS.get("rdp")().epsilon(
                    worst, plan.sample_rate, plan.steps, delta
                )
            except Exception:
                report["epsilon_worst_case"] = math.inf
        if self.model is not None:
            d = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            radial, angular = noise_scales(self.max_grad_norm, sigma, self.beta, d, plan.expected_batch_size)
            report["noise_std_radius"] = radial
            report["noise_std_angle"] = angular
        return report
