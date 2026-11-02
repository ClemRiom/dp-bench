"""Experiment configuration: YAML files -> typed, validated dataclasses.

A configuration describes a whole experiment: the dataset, the model, the mechanisms (each
with its own hyper-parameters), the privacy budgets, the seeds and the metrics.
It is read with OmegaConf, so any value can be overridden from the command line with a
dot-list (``training.epochs=2 seeds=[0] mechanisms.1.params.lr=0.1``). It is then validated:
unknown keys, unknown plugin names and unknown plugin parameters are reported with a
suggestion, before anything is trained.
"""

from __future__ import annotations

import difflib
import math
import numbers
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from dpbench.core.registry import ACCOUNTANTS, DATASETS, MECHANISMS, METRICS, MODELS, load_plugins

DEFAULT_METRICS = ("accuracy", "f1_macro", "roc_auc", "loss")


class ConfigError(ValueError):
    """The configuration is invalid."""


@dataclass
class PluginSpec:
    """A plugin name and the keyword arguments given to its constructor."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MechanismSpec:
    """A mechanism, its hyper-parameters, a display label and optional training overrides."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    training: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 256
    eval_batch_size: int = 4096


@dataclass
class PrivacyConfig:
    epsilons: list[float] = field(default_factory=lambda: [1.0, 4.0])
    delta: float | str = "auto"
    accountant: str | None = None

    def resolve_delta(self, n_train: int) -> float:
        """``auto`` means delta = 1/N, the convention of the internship plan."""
        return 1.0 / n_train if self.delta == "auto" else float(self.delta)


@dataclass
class ExperimentConfig:
    name: str
    dataset: PluginSpec
    model: PluginSpec
    mechanisms: list[MechanismSpec]
    description: str = ""
    seeds: list[int] = field(default_factory=lambda: [0])
    device: str = "cpu"
    plugins: list[str] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    metrics: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    output_dir: str = "runs"
    base_dir: str | None = field(default=None, repr=False, compare=False)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", self.name).strip("-") or "experiment"

    def training_for(self, mechanism: MechanismSpec) -> TrainingConfig:
        """The global training section with the mechanism's overrides applied."""
        return replace(self.training, **mechanism.training)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "seeds": list(self.seeds),
            "device": self.device,
            "plugins": list(self.plugins),
            "dataset": {"name": self.dataset.name, "params": dict(self.dataset.params)},
            "model": {"name": self.model.name, "params": dict(self.model.params)},
            "training": {f.name: getattr(self.training, f.name) for f in fields(TrainingConfig)},
            "privacy": {
                "epsilons": list(self.privacy.epsilons),
                "delta": self.privacy.delta,
                "accountant": self.privacy.accountant,
            },
            "mechanisms": [
                {"name": m.name, "label": m.label, "params": dict(m.params), "training": dict(m.training)}
                for m in self.mechanisms
            ],
            "metrics": list(self.metrics),
            "output_dir": self.output_dir,
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=1000)


_TOP_LEVEL = (
    "name",
    "description",
    "seeds",
    "device",
    "plugins",
    "dataset",
    "model",
    "training",
    "privacy",
    "mechanisms",
    "metrics",
    "output_dir",
)
_TRAINING_KEYS = tuple(f.name for f in fields(TrainingConfig))
_PRIVACY_KEYS = tuple(f.name for f in fields(PrivacyConfig))


def _check_keys(section: str, data: dict[str, Any], allowed: Sequence[str]) -> None:
    for key in data:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), list(allowed), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ConfigError(f"Unknown key {key!r} in {section}.{hint} Allowed keys: {', '.join(allowed)}.")


def _mapping(section: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a mapping, got {value!r}")
    return value


def _integer(section: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigError(f"{section} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{section} must be >= {minimum}, got {value}")
    return int(value)


def _number(section: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"{section} must be a number, got {value!r}")
    return float(value)


def _string_list(section: str, value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{section} must be a list of names, got {value!r}")
    return list(value)


def _plugin_spec(section: str, value: Any) -> PluginSpec:
    if isinstance(value, str):
        return PluginSpec(value)
    if isinstance(value, dict):
        _check_keys(section, value, ("name", "params"))
        if not isinstance(value.get("name"), str):
            raise ConfigError(f"{section} needs a 'name'")
        return PluginSpec(value["name"], dict(_mapping(f"{section}.params", value.get("params"))))
    raise ConfigError(f"{section} must be a plugin name or a mapping with 'name' and 'params', got {value!r}")


def _training_overrides(section: str, value: Any) -> dict[str, Any]:
    data = _mapping(section, value)
    _check_keys(section, data, _TRAINING_KEYS)
    return {key: _integer(f"{section}.{key}", v, 1) for key, v in data.items()}


def _mechanism_spec(index: int, value: Any) -> MechanismSpec:
    section = f"mechanisms[{index}]"
    if isinstance(value, str):
        return MechanismSpec(value)
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be a mechanism name or a mapping, got {value!r}")
    _check_keys(section, value, ("name", "label", "params", "training"))
    if not isinstance(value.get("name"), str):
        raise ConfigError(f"{section} needs a 'name'")
    label = value.get("label") or ""
    if not isinstance(label, str):
        raise ConfigError(f"{section}.label must be a string")
    return MechanismSpec(
        name=value["name"],
        params=dict(_mapping(f"{section}.params", value.get("params"))),
        label=label,
        training=_training_overrides(f"{section}.training", value.get("training")),
    )


def config_from_dict(data: Any, base_dir: str | Path | None = None) -> ExperimentConfig:
    """Build and validate a configuration from a plain dictionary (e.g. a parsed YAML file)."""
    if not isinstance(data, dict):
        raise ConfigError("The configuration must be a mapping")
    # Sprint 1 configurations had an "attacks" section; attacks are out of the scope of sprint 2.
    data = {key: value for key, value in data.items() if key != "attacks"}
    _check_keys("the configuration", data, _TOP_LEVEL)

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("The configuration needs a non-empty 'name'")
    for key in ("dataset", "model", "mechanisms"):
        if data.get(key) in (None, [], ""):
            raise ConfigError(f"The configuration needs a '{key}' section")

    training_data = _mapping("training", data.get("training"))
    _check_keys("training", training_data, _TRAINING_KEYS)
    training = TrainingConfig(**{key: _integer(f"training.{key}", value, 1) for key, value in training_data.items()})

    privacy_data = _mapping("privacy", data.get("privacy"))
    _check_keys("privacy", privacy_data, _PRIVACY_KEYS)
    epsilons_raw = privacy_data.get("epsilons", [1.0, 4.0])
    if isinstance(epsilons_raw, numbers.Real):
        epsilons_raw = [epsilons_raw]
    if not isinstance(epsilons_raw, list) or not epsilons_raw:
        raise ConfigError("privacy.epsilons must be a non-empty list of positive numbers")
    epsilons = sorted({_number("privacy.epsilons", e) for e in epsilons_raw})
    if any(e <= 0 or not math.isfinite(e) for e in epsilons):
        raise ConfigError(f"privacy.epsilons must be positive and finite, got {epsilons}")
    delta = privacy_data.get("delta", "auto")
    if delta != "auto":
        delta = _number("privacy.delta", delta)
        if not 0 < delta < 1:
            raise ConfigError(f"privacy.delta must be 'auto' or in (0, 1), got {delta}")
    accountant = privacy_data.get("accountant")
    if accountant is not None and not isinstance(accountant, str):
        raise ConfigError("privacy.accountant must be an accountant name")
    privacy = PrivacyConfig(epsilons=epsilons, delta=delta, accountant=accountant)

    mechanisms_raw = data["mechanisms"]
    if not isinstance(mechanisms_raw, list):
        mechanisms_raw = [mechanisms_raw]
    mechanisms = [_mechanism_spec(i, m) for i, m in enumerate(mechanisms_raw)]
    labels = [m.label for m in mechanisms]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ConfigError(f"Several mechanisms share the label(s) {duplicates}; give each one a distinct 'label'")

    seeds_raw = data.get("seeds", [0])
    if isinstance(seeds_raw, numbers.Integral) and not isinstance(seeds_raw, bool):
        seeds_raw = [seeds_raw]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ConfigError("seeds must be a non-empty list of integers")
    seeds = [_integer("seeds", s, 0) for s in seeds_raw]
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"seeds contains duplicates: {seeds}")

    device = data.get("device", "cpu")
    if not isinstance(device, str):
        raise ConfigError("device must be a string such as 'cpu', 'cuda' or 'auto'")
    description = data.get("description") or ""
    output_dir = data.get("output_dir") or "runs"
    if not isinstance(description, str) or not isinstance(output_dir, str):
        raise ConfigError("description and output_dir must be strings")

    return ExperimentConfig(
        name=name.strip(),
        description=description.strip(),
        dataset=_plugin_spec("dataset", data["dataset"]),
        model=_plugin_spec("model", data["model"]),
        mechanisms=mechanisms,
        seeds=seeds,
        device=device,
        plugins=_string_list("plugins", data.get("plugins")),
        training=training,
        privacy=privacy,
        metrics=_string_list("metrics", data.get("metrics", list(DEFAULT_METRICS))),
        output_dir=output_dir,
        base_dir=str(base_dir) if base_dir is not None else None,
    )


def validate_plugins(config: ExperimentConfig) -> None:
    """Load the workspace and the configuration's external plugins, then check every plugin name and parameter."""
    from dpbench.workspace import load_workspace

    load_workspace()  # datasets and algorithms added in the dashboard
    try:
        load_plugins(config.plugins, config.base_dir)
        DATASETS.check_params(config.dataset.name, config.dataset.params)
        MODELS.check_params(config.model.name, config.model.params)
        for mechanism in config.mechanisms:
            MECHANISMS.check_params(mechanism.name, mechanism.params)
            accountant = mechanism.params.get("accountant")
            if accountant is not None:
                ACCOUNTANTS.get(accountant)
        for metric in config.metrics:
            METRICS.get(metric)
        if config.privacy.accountant is not None:
            ACCOUNTANTS.get(config.privacy.accountant)
    except (LookupError, ValueError, FileNotFoundError, ImportError) as error:
        raise ConfigError(str(error)) from error


def parse_overrides(overrides: Iterable[str]) -> list[tuple[str, Any]]:
    """``["training.epochs=2", "seeds=[0, 1]"]`` -> ``[("training.epochs", 2), ("seeds", [0, 1])]``."""
    parsed = []
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override {item!r} must look like key=value (e.g. training.epochs=2)")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Override {item!r} has an empty key")
        try:
            value = yaml.safe_load(raw) if raw.strip() else None
        except yaml.YAMLError as error:
            raise ConfigError(f"Cannot parse the value of override {item!r}: {error}") from error
        parsed.append((key, value))
    return parsed


def load_config(path: str | Path, overrides: Iterable[str] = (), validate: bool = True) -> ExperimentConfig:
    """Read a YAML configuration, apply dot-list overrides and validate it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        cfg = OmegaConf.load(path)
        for key, value in parse_overrides(overrides):
            OmegaConf.update(cfg, key, value, force_add=True)
        data = OmegaConf.to_container(cfg, resolve=True)
    except ConfigError:
        raise
    except Exception as error:  # OmegaConf raises many exception types
        raise ConfigError(f"Cannot read {path}: {error}") from error
    config = config_from_dict(data, base_dir=path.resolve().parent)
    if validate:
        validate_plugins(config)
    return config


def load_config_text(text: str, base_dir: str | Path | None = None, validate: bool = True) -> ExperimentConfig:
    """Parse a configuration given as YAML text (used by the dashboard)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML: {error}") from error
    config = config_from_dict(data, base_dir=base_dir)
    if validate:
        validate_plugins(config)
    return config
