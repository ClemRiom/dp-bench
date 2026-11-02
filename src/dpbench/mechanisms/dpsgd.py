"""DP-SGD (Abadi et al. 2016) with the in-house engine."""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from dpbench.accounting.base import Accountant
from dpbench.accounting.calibration import calibrate_noise
from dpbench.core.registry import register_mechanism
from dpbench.core.seeding import derive_seed, torch_generator
from dpbench.core.types import FitResult
from dpbench.mechanisms.base import (
    OPTIMIZERS,
    EpochCallback,
    Mechanism,
    SamplingPlan,
    make_optimizer,
    summarize_epoch,
    to_tensors,
)
from dpbench.mechanisms.engine import PrivateGradient


@register_mechanism("dpsgd")
class DPSGD(Mechanism):
    """DP-SGD (Abadi et al. 2016): Poisson sampling, per-example L2 clipping, Gaussian noise; in-house engine built on torch.func.

    Before training, the noise multiplier is calibrated: the smallest one for which the
    accountant certifies the target epsilon after the last step. After training, the
    accountant recomputes the epsilon actually spent. Subclasses change the noise, the norm
    of the clipping or the optimizer through class attributes, and the private gradient
    itself through ``engine_class`` (a subclass of :class:`PrivateGradient`, see GeoDP).
    """

    title = "DP-SGD"
    noise = "gaussian"
    clipping_norm: ClassVar[int] = 2
    default_accountant = "prv"
    default_optimizer: ClassVar[str] = "sgd"
    default_lr: ClassVar[float] = 0.1
    #: Number of epochs at which the epsilon spent so far is recorded in the history.
    history_epsilon_points: ClassVar[int] = 12
    #: The class that computes the private gradient of a batch.
    engine_class: ClassVar[type[PrivateGradient]] = PrivateGradient

    def __init__(
        self,
        lr: float | None = None,
        max_grad_norm: float = 1.0,
        optimizer: str | None = None,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        accountant: str | None = None,
        noise_multiplier: float | None = None,
        max_physical_batch_size: int = 512,
    ) -> None:
        super().__init__()
        self.lr = float(self.default_lr if lr is None else lr)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = optimizer or self.default_optimizer
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer {self.optimizer!r}; choose one of {OPTIMIZERS}")
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.accountant = accountant
        self.noise_multiplier = None if noise_multiplier is None else float(noise_multiplier)
        self.max_physical_batch_size = int(max_physical_batch_size)

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "lr": cls.default_lr,
            "max_grad_norm": 1.0,
            "optimizer": cls.default_optimizer,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "accountant": cls.default_accountant,
            "noise_multiplier": "calibrated for the target epsilon",
            "max_physical_batch_size": 512,
        }

    def fit(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        *,
        target_epsilon: float,
        delta: float,
        epochs: int,
        batch_size: int,
        seed: int,
        device: str | torch.device = "cpu",
        accountant: str | None = None,
        callback: EpochCallback | None = None,
    ) -> FitResult:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        plan = SamplingPlan(len(y), batch_size, epochs)
        privacy_accountant = self.resolve_accountant(self.accountant, accountant)

        start = time.perf_counter()
        if self.noise_multiplier is None:
            accounted = calibrate_noise(
                privacy_accountant, target_epsilon, plan.sample_rate, plan.steps, delta, self.noise or "gaussian"
            )
            sigma = self.noise_multiplier_for(accounted, plan)
        else:
            sigma = self.noise_multiplier
        calibration_time = time.perf_counter() - start

        history, extra = self._train(plan, X, y, sigma, privacy_accountant, delta, seed, callback)
        epsilon = self._epsilon_after(privacy_accountant, sigma, plan, plan.steps, delta)
        if self.noise_multiplier is None and epsilon > target_epsilon * (1 + 1e-9):
            raise RuntimeError(f"Accounting check failed: spent epsilon {epsilon:.6f} > target {target_epsilon}")
        extra.update(self.privacy_report(privacy_accountant, sigma, plan, delta))
        extra["calibration_time_s"] = calibration_time
        self.fit_result = FitResult(
            epsilon=epsilon,
            delta=delta,
            noise_multiplier=sigma,
            sample_rate=plan.sample_rate,
            steps=plan.steps,
            epochs=epochs,
            batch_size=batch_size,
            accountant=privacy_accountant.name,
            train_time_s=float(sum(row["seconds"] for row in history)),
            history=history,
            extra=extra,
        )
        return self.fit_result

    def accounted_noise_multiplier(self, sigma: float, plan: SamplingPlan) -> float:
        """The noise multiplier the accountant analyses when the engine uses ``sigma`` (the same for DP-SGD)."""
        return sigma

    def noise_multiplier_for(self, accounted: float, plan: SamplingPlan) -> float:
        """The engine's noise multiplier for an accounted one: the inverse of :meth:`accounted_noise_multiplier`."""
        return accounted

    def privacy_report(self, accountant: Accountant, sigma: float, plan: SamplingPlan, delta: float) -> dict[str, Any]:
        """Extra privacy columns of the results (none for DP-SGD)."""
        return {}

    def engine_options(self) -> dict[str, Any]:
        """Extra keyword arguments of ``engine_class`` (none for DP-SGD)."""
        return {}

    def _epsilon_after(
        self, accountant: Accountant, sigma: float, plan: SamplingPlan, steps: int, delta: float
    ) -> float:
        accounted = self.accounted_noise_multiplier(sigma, plan)
        return accountant.epsilon(accounted, plan.sample_rate, steps, delta, self.noise or "gaussian")

    def _train(
        self,
        plan: SamplingPlan,
        X: np.ndarray,
        y: np.ndarray,
        sigma: float,
        accountant: Accountant,
        delta: float,
        seed: int,
        callback: EpochCallback | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        assert self.model is not None
        X_t, y_t = to_tensors(X, y, self.device)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer, self.lr, self.momentum, self.weight_decay)
        engine = self.engine_class(
            self.model,
            max_grad_norm=self.max_grad_norm,
            noise_multiplier=sigma,
            noise=self.noise or "gaussian",
            norm=self.clipping_norm,
            max_physical_batch_size=self.max_physical_batch_size,
            generator=torch_generator(derive_seed(seed, "noise"), self.device),
            **self.engine_options(),
        )
        sampler = torch_generator(derive_seed(seed, "sampling"))
        history = []
        self.model.train()
        for epoch in range(1, plan.epochs + 1):
            start = time.perf_counter()
            stats = []
            for _ in range(plan.steps_per_epoch):
                index = (torch.rand(plan.n, generator=sampler) < plan.sample_rate).nonzero(as_tuple=True)[0]
                index = index.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                stats.append(engine.privatize(X_t[index], y_t[index], plan.expected_batch_size))
                optimizer.step()
            seconds = time.perf_counter() - start
            epsilon = self._epsilon_so_far(accountant, sigma, plan, epoch, delta)
            row = summarize_epoch(epoch, stats, seconds, epsilon)
            history.append(row)
            if callback is not None:
                callback(row)
        return history, {}

    def _epsilon_so_far(
        self, accountant: Accountant, sigma: float, plan: SamplingPlan, epoch: int, delta: float
    ) -> float:
        """Epsilon after ``epoch`` epochs, computed at ``history_epsilon_points`` evenly spaced epochs (NaN elsewhere)."""
        points = max(1, min(plan.epochs, self.history_epsilon_points))
        recorded = {round(plan.epochs * k / points) for k in range(1, points + 1)}
        if epoch not in recorded:
            return float("nan")
        return self._epsilon_after(accountant, sigma, plan, epoch * plan.steps_per_epoch, delta)
