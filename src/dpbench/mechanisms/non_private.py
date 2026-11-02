"""Non-private training: the utility baseline."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dpbench.core.registry import register_mechanism
from dpbench.core.seeding import derive_seed, torch_generator
from dpbench.core.types import FitResult
from dpbench.mechanisms.base import OPTIMIZERS, EpochCallback, Mechanism, make_optimizer, summarize_epoch, to_tensors


@register_mechanism("non_private")
class NonPrivate(Mechanism):
    """Standard mini-batch training without differential privacy: the utility baseline (epsilon = infinity)."""

    title = "Non-private"
    private = False

    def __init__(
        self, lr: float = 0.1, optimizer: str = "sgd", momentum: float = 0.0, weight_decay: float = 0.0
    ) -> None:
        super().__init__()
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer {optimizer!r}; choose one of {OPTIMIZERS}")
        self.lr = float(lr)
        self.optimizer = optimizer
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

    def fit(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        *,
        target_epsilon: float = math.inf,
        delta: float = 0.0,
        epochs: int,
        batch_size: int,
        seed: int,
        device: str | torch.device = "cpu",
        accountant: str | None = None,
        callback: EpochCallback | None = None,
    ) -> FitResult:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        X_t, y_t = to_tensors(X, y, self.device)
        optimizer = make_optimizer(self.model.parameters(), self.optimizer, self.lr, self.momentum, self.weight_decay)
        shuffler = torch_generator(derive_seed(seed, "shuffle"))
        n = len(y)
        history = []
        self.model.train()
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            stats = []
            order = torch.randperm(n, generator=shuffler).to(self.device)
            for begin in range(0, n, batch_size):
                index = order[begin : begin + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model(X_t[index]), y_t[index])
                loss.backward()
                optimizer.step()
                stats.append({"examples": float(len(index)), "loss_sum": loss.detach().item() * len(index)})
            row = summarize_epoch(epoch, stats, time.perf_counter() - start, math.inf)
            history.append(row)
            if callback is not None:
                callback(row)
        self.fit_result = FitResult(
            epsilon=math.inf,
            delta=delta,
            steps=epochs * math.ceil(n / batch_size),
            epochs=epochs,
            batch_size=batch_size,
            train_time_s=float(sum(row["seconds"] for row in history)),
            history=history,
        )
        return self.fit_result
