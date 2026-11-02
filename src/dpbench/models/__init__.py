"""Model plugins: PyTorch modules built from the input shape and the number of classes.

None of them uses BatchNorm: it mixes the examples of a batch, so per-example gradients
(and hence DP-SGD) are not defined for it. GroupNorm is used instead where normalisation helps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from dpbench.core.registry import register_model

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


def make_activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; choose one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


def _as_tuple(value: int | Sequence[int]) -> tuple[int, ...]:
    return (int(value),) if isinstance(value, int) else tuple(int(v) for v in value)


@register_model("mlp")
class MLP(nn.Module):
    """Fully connected network with configurable hidden layers and activation."""

    def __init__(
        self,
        input_shape: Sequence[int],
        n_classes: int,
        hidden: int | Sequence[int] = (64,),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        width = math.prod(input_shape)
        for size in _as_tuple(hidden):
            layers += [nn.Linear(width, size), make_activation(activation)]
            width = size
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _image_shape(input_shape: Sequence[int], model: str) -> tuple[int, int, int]:
    if len(input_shape) != 3:
        raise ValueError(f"{model} expects images (channels, height, width), got input shape {tuple(input_shape)}")
    return tuple(int(s) for s in input_shape)  # type: ignore[return-value]


@register_model("cnn")
class CNN(nn.Module):
    """Small CNN of Papernot et al. (2021) with tanh activations, for 28x28 images (about 26,000 parameters)."""

    def __init__(
        self,
        input_shape: Sequence[int],
        n_classes: int,
        channels: Sequence[int] = (16, 32),
        hidden: int = 32,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        shape = _image_shape(input_shape, "cnn")
        c1, c2 = _as_tuple(channels)
        self.features = nn.Sequential(
            nn.Conv2d(shape[0], c1, kernel_size=8, stride=2, padding=3),
            make_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2),
            make_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )
        with torch.no_grad():
            width = self.features(torch.zeros(1, *shape)).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(width, hidden), make_activation(activation), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@register_model("cnn_gn")
class GroupNormCNN(nn.Module):
    """CNN with GroupNorm for 32x32 colour images: three blocks of two 3x3 convolutions, then a linear layer."""

    def __init__(
        self,
        input_shape: Sequence[int],
        n_classes: int,
        channels: Sequence[int] = (32, 64, 128),
        groups: int = 8,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        shape = _image_shape(input_shape, "cnn_gn")
        layers: list[nn.Module] = []
        width = shape[0]
        for out in _as_tuple(channels):
            for _ in range(2):
                layers += [
                    nn.Conv2d(width, out, kernel_size=3, padding=1),
                    nn.GroupNorm(math.gcd(groups, out), out),
                    make_activation(activation),
                ]
                width = out
            layers.append(nn.AvgPool2d(kernel_size=2))
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
