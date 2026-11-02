"""Interface of the training algorithms (the "mechanisms") and shared training helpers."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn

from dpbench.accounting.base import Accountant, UnsupportedNoiseError
from dpbench.core.registry import ACCOUNTANTS
from dpbench.core.types import FitResult

logger = logging.getLogger(__name__)

EpochCallback = Callable[[dict[str, float]], None]

OPTIMIZERS = ("sgd", "adam", "adamw")


def make_optimizer(
    parameters: Iterable[nn.Parameter], name: str, lr: float, momentum: float = 0.0, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; choose one of {OPTIMIZERS}")


@dataclass(frozen=True)
class SamplingPlan:
    """How many steps a training makes and with which Poisson sampling rate.

    As in Opacus, an epoch is ``ceil(N / batch_size)`` steps and the sampling rate is the
    inverse of that number, so the expected batch size is close to ``batch_size``.
    """

    n: int
    batch_size: int
    epochs: int

    @property
    def steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.n / self.batch_size))

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.steps_per_epoch

    @property
    def steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def expected_batch_size(self) -> int:
        return max(1, int(self.n * self.sample_rate))


class Mechanism(ABC):
    """Base class of the mechanism plugins.

    The runner calls :meth:`fit` once per (target epsilon, seed) with a freshly initialised
    model, then evaluates it with :meth:`predict_proba`. The privacy-related attributes are
    class variables, so a variant can be declared in a few lines (see ``dp_lap_sgd.py``).
    """

    name: ClassVar[str]
    #: Display name in tables and figures (defaults to the registered name).
    title: ClassVar[str] = ""
    #: False for the non-private baseline (trained once per seed, epsilon = infinity).
    private: ClassVar[bool] = True
    #: Kind of noise ("gaussian" or "laplace"), used to pick a compatible accountant.
    noise: ClassVar[str | None] = None
    default_accountant: ClassVar[str | None] = None

    def __init__(self) -> None:
        self.model: nn.Module | None = None
        self.device = torch.device("cpu")
        self.fit_result: FitResult | None = None

    @abstractmethod
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
        """Train ``model`` on ``(X, y)`` within the budget ``(target_epsilon, delta)``."""

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Class probabilities (float64) of the trained model."""
        if self.model is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        model = self.model
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(X[start : start + batch_size])).to(self.device)
                outputs.append(torch.softmax(model(batch).double(), dim=1).cpu().numpy())
        if not outputs:
            return np.zeros((0, 0))
        return np.concatenate(outputs)

    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return self.predict_proba(X, batch_size).argmax(axis=1)

    def get_privacy_spent(self) -> tuple[float, float]:
        """(epsilon, delta) actually spent by the last call to :meth:`fit`."""
        if self.fit_result is None:
            raise RuntimeError("fit() must be called first")
        return self.fit_result.epsilon, self.fit_result.delta

    @classmethod
    def resolve_accountant(cls, explicit: str | None, default: str | None) -> Accountant:
        """The accountant to use: the mechanism's own choice, else the experiment's, else the class default.

        The experiment-wide default is skipped (with a warning) when it cannot analyse this
        mechanism's noise, e.g. RDP for Laplace noise; an explicit choice raises instead.
        """
        noise = cls.noise or "gaussian"
        if explicit is not None:
            accountant = ACCOUNTANTS.get(explicit)()
            accountant.check(noise)
            return accountant
        if default is not None:
            accountant = ACCOUNTANTS.get(default)()
            try:
                accountant.check(noise)
                return accountant
            except UnsupportedNoiseError:
                logger.warning(
                    "accountant %r cannot analyse %s noise; %s uses %r instead",
                    default,
                    noise,
                    cls.name,
                    cls.default_accountant,
                )
        if cls.default_accountant is None:
            raise ValueError(f"mechanism {cls.name!r} has no default accountant")
        accountant = ACCOUNTANTS.get(cls.default_accountant)()
        accountant.check(noise)
        return accountant


def to_tensors(X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)).to(device),
    )


def summarize_epoch(epoch: int, stats: list[dict[str, float]], seconds: float, epsilon: float) -> dict[str, Any]:
    """One row of the training history from the statistics of the epoch's steps."""
    examples = sum(s["examples"] for s in stats)
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": sum(s["loss_sum"] for s in stats) / examples if examples else math.nan,
        "epsilon": epsilon,
        "seconds": seconds,
        "mean_batch_size": examples / len(stats) if stats else 0.0,
    }
    if stats and "clipped" in stats[0]:
        row["clipped_fraction"] = sum(s["clipped"] for s in stats) / examples if examples else math.nan
        row["mean_grad_norm"] = sum(s["norm_sum"] for s in stats) / examples if examples else math.nan
    return row
