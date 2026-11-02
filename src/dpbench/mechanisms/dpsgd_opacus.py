"""DP-SGD with Opacus, the reference implementation used to cross-check the in-house engine."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dpbench.accounting.base import Accountant
from dpbench.core.registry import register_mechanism
from dpbench.core.seeding import derive_seed, torch_generator
from dpbench.mechanisms.base import EpochCallback, SamplingPlan, make_optimizer, summarize_epoch, to_tensors
from dpbench.mechanisms.dpsgd import DPSGD

OPACUS_ACCOUNTANTS = ("rdp", "prv", "gdp")


@register_mechanism("dpsgd_opacus")
class OpacusDPSGD(DPSGD):
    """DP-SGD with Opacus (Meta's reference library): the same algorithm as dpsgd, used to cross-check the in-house engine.

    The noise multiplier is calibrated by DP-Bench's accountant exactly as for ``dpsgd``;
    Opacus' own accountant is also queried after training and its epsilon is reported in
    the ``epsilon_opacus`` column. ``secure_mode`` requires the ``torchcsprng`` package.
    """

    title = "DP-SGD (Opacus)"

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
        secure_mode: bool = False,
    ) -> None:
        super().__init__(
            lr, max_grad_norm, optimizer, momentum, weight_decay, accountant, noise_multiplier, max_physical_batch_size
        )
        self.secure_mode = bool(secure_mode)

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {**super().default_params(), "secure_mode": False}

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
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager

        assert self.model is not None
        X_t, y_t = to_tensors(X, y, torch.device("cpu"))
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=plan.batch_size)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer, self.lr, self.momentum, self.weight_decay)
        opacus_accountant = accountant.name if accountant.name in OPACUS_ACCOUNTANTS else "prv"
        engine = PrivacyEngine(accountant=opacus_accountant, secure_mode=self.secure_mode)
        torch.manual_seed(derive_seed(seed, "sampling"))  # Opacus draws the Poisson batches from the global generator
        noise_generator = None if self.secure_mode else torch_generator(derive_seed(seed, "noise"), self.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            private_model, private_optimizer, private_loader = engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=sigma,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
                noise_generator=noise_generator,
            )
        history = []
        try:
            for epoch in range(1, plan.epochs + 1):
                private_model.train()
                start = time.perf_counter()
                loss_sum, examples = 0.0, 0
                with (
                    warnings.catch_warnings(),
                    BatchMemoryManager(
                        data_loader=private_loader,
                        max_physical_batch_size=self.max_physical_batch_size,
                        optimizer=private_optimizer,
                    ) as physical_loader,
                ):
                    warnings.simplefilter("ignore")
                    for x_batch, y_batch in physical_loader:
                        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                        private_optimizer.zero_grad()
                        loss = F.cross_entropy(private_model(x_batch), y_batch)
                        loss.backward()
                        private_optimizer.step()
                        if len(y_batch):
                            loss_sum += loss.detach().item() * len(y_batch)
                            examples += len(y_batch)
                seconds = time.perf_counter() - start
                epsilon = self._epsilon_so_far(accountant, sigma, plan, epoch, delta)
                row = summarize_epoch(epoch, [{"examples": float(examples), "loss_sum": loss_sum}], seconds, epsilon)
                row["mean_batch_size"] = examples / plan.steps_per_epoch
                history.append(row)
                if callback is not None:
                    callback(row)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                epsilon_opacus = float(engine.get_epsilon(delta))
        finally:
            close = getattr(private_model, "_close", None) or getattr(private_model, "remove_hooks", None)
            if close is not None:
                close()
        return history, {"epsilon_opacus": epsilon_opacus, "opacus_accountant": opacus_accountant}
