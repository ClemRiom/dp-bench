"""Layout of a run directory and helpers to read it back (used by the report and the dashboard)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CONFIG = "config.yaml"
ENVIRONMENT = "environment.json"
DATASET = "dataset.json"
RESULTS = "results.csv"
SUMMARY = "summary.csv"
HISTORY = "history.csv"
STATUS = "status.json"
LOG = "run.log"


def write_json(path: Path, data: Any) -> None:
    def default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=default, allow_nan=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def is_run_directory(path: Path) -> bool:
    return (path / CONFIG).is_file() and ((path / RESULTS).is_file() or (path / STATUS).is_file())


def run_kind(path: Path) -> str:
    """``"classic"`` for a classic-DP benchmark (queries on a table), ``"ml"`` for a training experiment."""
    try:
        config = yaml.safe_load((path / CONFIG).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return "ml"
    return str(config.get("kind", "ml")) if isinstance(config, dict) else "ml"


def find_runs(roots: list[str | Path], kind: str | None = "ml") -> list[Path]:
    """Run directories of the given kind (``None`` = all) in (or one level below) the roots, most recent first."""
    found: dict[Path, float] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        candidates = [root, *root.iterdir()]
        candidates += [child for c in list(candidates[1:]) if c.is_dir() for child in c.iterdir()]
        for candidate in candidates:
            if candidate.is_dir() and is_run_directory(candidate) and (kind is None or run_kind(candidate) == kind):
                found[candidate.resolve()] = _started(candidate)
    return sorted(found, key=lambda p: found[p], reverse=True)


def _started(path: Path) -> float:
    status = read_json(path / STATUS)
    try:
        return datetime.fromisoformat(status["started"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return (path / CONFIG).stat().st_mtime


@dataclass
class Run:
    """Everything a run directory contains, loaded in memory."""

    path: Path
    config: dict[str, Any]
    dataset: dict[str, Any]
    environment: dict[str, Any]
    status: dict[str, Any]
    results: pd.DataFrame
    history: pd.DataFrame

    @property
    def name(self) -> str:
        return str(self.config.get("name", self.path.name))

    @property
    def started(self) -> str:
        return str(self.status.get("started", ""))[:19].replace("T", " ")

    @property
    def ok(self) -> pd.DataFrame:
        if self.results.empty or "status" not in self.results:
            return self.results
        return self.results[self.results["status"] == "ok"]

    @property
    def mechanisms(self) -> list[str]:
        """Mechanism labels in the order of the configuration."""
        labels = [m.get("label") or m["name"] for m in self.config.get("mechanisms", [])]
        present = set(self.results["mechanism"]) if "mechanism" in self.results else set()
        return [label for label in labels if label in present]

    @property
    def private_mechanisms(self) -> list[str]:
        ok = self.ok
        if ok.empty:
            return []
        private = set(ok.loc[ok["private"].astype(bool), "mechanism"])
        return [m for m in self.mechanisms if m in private]

    @property
    def epsilons(self) -> list[float]:
        ok = self.ok
        if ok.empty:
            return []
        return sorted(e for e in ok["epsilon_target"].unique() if math.isfinite(e))

    def summary(self) -> pd.DataFrame:
        from dpbench.experiments.aggregate import summarize

        return summarize(self.results, self.mechanisms)

    def label(self) -> str:
        dataset = self.dataset.get("name", "?")
        state = self.status.get("state", "")
        suffix = "" if state in ("finished", "") else f" [{state}]"
        return f"{self.name} - {dataset} - {self.started}{suffix}"


def load_run(path: str | Path) -> Run:
    path = Path(path)
    if not is_run_directory(path):
        raise FileNotFoundError(f"{path} is not a DP-Bench run directory")
    config = yaml.safe_load((path / CONFIG).read_text(encoding="utf-8")) or {}
    return Run(
        path=path,
        config=config,
        dataset=read_json(path / DATASET),
        environment=read_json(path / ENVIRONMENT),
        status=read_json(path / STATUS),
        results=read_csv(path / RESULTS),
        history=read_csv(path / HISTORY),
    )
