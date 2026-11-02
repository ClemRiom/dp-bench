"""Experiment runner: every mechanism x target epsilon x seed, evaluated with every metric.

Results are written to the run directory after each training, so a running experiment can
be followed from the dashboard, and a failed training is recorded (with its error) without
stopping the others.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from dpbench.core.config import ExperimentConfig, MechanismSpec, validate_plugins
from dpbench.core.environment import environment_snapshot
from dpbench.core.registry import DATASETS, MECHANISMS, METRICS, MODELS, plugin_file
from dpbench.core.seeding import derive_seed, resolve_device, seed_everything
from dpbench.core.types import Dataset
from dpbench.experiments import storage
from dpbench.experiments.aggregate import format_mean_std, summarize
from dpbench.models import count_parameters

TRAIN_EVAL_SIZE = 10_000


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _eps_text(epsilon: float) -> str:
    return "inf" if math.isinf(epsilon) else f"{epsilon:g}"


@dataclass(frozen=True)
class Job:
    """One training: a mechanism, a target epsilon and a seed."""

    index: int
    mechanism: MechanismSpec
    epsilon: float
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.mechanism.label}|eps={_eps_text(self.epsilon)}|seed={self.seed}"


def plan_jobs(config: ExperimentConfig) -> list[Job]:
    """All trainings of an experiment. The non-private baseline is trained once per seed."""
    jobs: list[Job] = []
    for spec in config.mechanisms:
        epsilons = config.privacy.epsilons if MECHANISMS.get(spec.name).private else [math.inf]
        for epsilon in epsilons:
            for seed in config.seeds:
                jobs.append(Job(len(jobs) + 1, spec, float(epsilon), seed))
    return jobs


@dataclass
class ExperimentResult:
    directory: Path
    results: pd.DataFrame
    summary: pd.DataFrame
    history: pd.DataFrame


def run_job(
    config: ExperimentConfig,
    job: Job,
    dataset: Dataset,
    delta: float,
    device: torch.device,
    log: logging.Logger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train and evaluate one model. Returns the result row and the training history."""
    spec = job.mechanism
    mechanism_cls = MECHANISMS.get(spec.name)
    training = config.training_for(spec)
    row: dict[str, Any] = {
        "run_id": job.run_id,
        "mechanism": spec.label,
        "mechanism_name": spec.name,
        "private": bool(mechanism_cls.private),
        "epsilon_target": job.epsilon,
        "seed": job.seed,
        "status": "ok",
        "error": "",
        "delta": delta,
        "epochs": training.epochs,
        "batch_size": training.batch_size,
    }
    history: list[dict[str, Any]] = []
    try:
        seed_everything(derive_seed(job.seed, "model"))
        model = MODELS.get(config.model.name)(dataset.input_shape, dataset.n_classes, **config.model.params)
        row["n_parameters"] = count_parameters(model)
        mechanism = mechanism_cls(**spec.params)
        seed_everything(derive_seed(job.seed, "training", spec.label, job.epsilon))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fit = mechanism.fit(
            model,
            dataset.X_train,
            dataset.y_train,
            target_epsilon=job.epsilon,
            delta=delta,
            epochs=training.epochs,
            batch_size=training.batch_size,
            seed=job.seed,
            device=device,
            accountant=config.privacy.accountant,
        )
        row.update(
            epsilon_spent=fit.epsilon,
            accountant=fit.accountant or "",
            noise_multiplier=math.nan if fit.noise_multiplier is None else fit.noise_multiplier,
            sample_rate=math.nan if fit.sample_rate is None else fit.sample_rate,
            steps=fit.steps,
            train_time_s=fit.train_time_s,
            time_per_epoch_s=fit.train_time_s / max(fit.epochs, 1),
        )
        for key, value in fit.extra.items():
            if isinstance(value, (bool, int, float, str)):
                row[key] = value
        if device.type == "cuda":
            row["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20

        start = time.perf_counter()
        proba = mechanism.predict_proba(dataset.X_test, training.eval_batch_size)
        for name in config.metrics:
            row[f"test_{name}"] = METRICS.get(name)()(dataset.y_test, proba, dataset.groups_test)
        rng = np.random.default_rng(derive_seed(job.seed, "train-evaluation"))
        subset = rng.choice(dataset.n_train, size=min(TRAIN_EVAL_SIZE, dataset.n_train), replace=False)
        train_accuracy = float(np.mean(mechanism.predict(dataset.X_train[subset]) == dataset.y_train[subset]))
        row["train_accuracy"] = train_accuracy
        row["generalization_gap"] = train_accuracy - float(np.mean(proba.argmax(axis=1) == dataset.y_test))
        row["eval_time_s"] = time.perf_counter() - start
        history = [
            {"run_id": job.run_id, "mechanism": spec.label, "epsilon_target": job.epsilon, "seed": job.seed, **h}
            for h in fit.history
        ]
    except Exception as error:  # a failed training must not stop the experiment
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"
        log.error("%s failed: %s\n%s", job.run_id, row["error"], traceback.format_exc())
    return row, history


def _progress_line(job: Job, total: int, row: dict[str, Any]) -> str:
    head = f"[{job.index:>3}/{total}] {job.mechanism.label:<16} eps={_eps_text(job.epsilon):<6} seed={job.seed:<3}"
    if row["status"] != "ok":
        return f"{head} FAILED: {row['error']}"
    parts = [head]
    if "test_accuracy" in row:
        parts.append(f"accuracy={row['test_accuracy']:.4f}")
    if row["private"]:
        parts.append(f"eps_spent={row['epsilon_spent']:.3f} sigma={row['noise_multiplier']:.3f} ({row['accountant']})")
    parts.append(f"{row['train_time_s']:.1f}s")
    return "  ".join(parts)


def summary_table(summary: pd.DataFrame) -> str:
    """Plain-text summary: one line per (mechanism, epsilon)."""
    if summary.empty:
        return "(no successful run)"
    columns = [
        ("Accuracy", "test_accuracy"),
        ("eps spent", "epsilon_spent"),
        ("sigma", "noise_multiplier"),
        ("s/epoch", "time_per_epoch_s"),
    ]
    lines = []
    for _, r in summary.iterrows():
        line = {"Mechanism": r["mechanism"], "eps": _eps_text(r["epsilon_target"]), "seeds": int(r["n_seeds"])}
        for title, column in columns:
            if f"{column}_mean" in r:
                digits = 2 if column == "time_per_epoch_s" else 3
                line[title] = format_mean_std(r[f"{column}_mean"], r[f"{column}_std"], digits)
        lines.append(line)
    return pd.DataFrame(lines).to_string(index=False)


def portable_config(config: ExperimentConfig) -> str:
    """The resolved configuration as YAML, with plugin files relative to the current directory.

    A run directory can then be moved (for example into ``results/``) and re-run from the root
    of the checkout with ``dpbench run <run directory>/config.yaml``.
    """
    data = config.to_dict()
    plugins = []
    for spec in config.plugins:
        try:
            path = plugin_file(spec, Path(config.base_dir) if config.base_dir else None)
        except FileNotFoundError:
            path = None
        if path is None:
            plugins.append(spec)
            continue
        try:
            plugins.append(path.relative_to(Path.cwd().resolve()).as_posix())
        except ValueError:
            plugins.append(str(path))
    data["plugins"] = plugins
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000)


def _create_run_directory(root: Path, slug: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path, suffix = root / f"{stamp}_{slug}", 1
    while path.exists():
        suffix += 1
        path = root / f"{stamp}_{slug}-{suffix}"
    path.mkdir(parents=True)
    return path


def _attach_handlers(run_dir: Path, console: bool) -> list[logging.Handler]:
    package_logger = logging.getLogger("dpbench")
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False  # some libraries (Opacus) install a root handler
    file_handler = logging.FileHandler(run_dir / storage.LOG, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    handlers: list[logging.Handler] = [file_handler]
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(console_handler)
    for handler in handlers:
        package_logger.addHandler(handler)
    return handlers


def _detach_handlers(handlers: list[logging.Handler]) -> None:
    package_logger = logging.getLogger("dpbench")
    for handler in handlers:
        package_logger.removeHandler(handler)
        handler.close()
    if not package_logger.handlers:
        package_logger.propagate = True


def run_experiment(
    config: ExperimentConfig,
    output_dir: str | Path | None = None,
    *,
    run_dir: str | Path | None = None,
    console: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> ExperimentResult:
    """Run a whole experiment and write it to ``<output_dir>/<timestamp>_<name>/`` (or to ``run_dir``)."""
    validate_plugins(config)
    if run_dir is not None:
        run_dir = Path(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} already exists and is not empty")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        root = Path(output_dir if output_dir is not None else config.output_dir)
        run_dir = _create_run_directory(root, config.slug)
    handlers = _attach_handlers(run_dir, console)
    log = logging.getLogger("dpbench.run")
    jobs = plan_jobs(config)
    status: dict[str, Any] = {
        "state": "running",
        "started": _now(),
        "updated": _now(),
        "total": len(jobs),
        "done": 0,
        "failed": 0,
        "current": None,
        "pid": os.getpid(),
    }
    clock = time.perf_counter()
    try:
        (run_dir / storage.CONFIG).write_text(portable_config(config), encoding="utf-8")
        storage.write_json(run_dir / storage.STATUS, status)
        storage.write_json(run_dir / storage.ENVIRONMENT, environment_snapshot())
        log.info("DP-Bench experiment %r: %d trainings -> %s", config.name, len(jobs), run_dir)

        start = time.perf_counter()
        dataset = DATASETS.get(config.dataset.name)(**config.dataset.params).load()
        delta = config.privacy.resolve_delta(dataset.n_train)
        storage.write_json(
            run_dir / storage.DATASET,
            {**dataset.summary(), "delta": delta, "load_time_s": time.perf_counter() - start},
        )
        log.info(
            "Dataset %s: %d training / %d test examples, input %s, %d classes; delta = %.3g",
            dataset.name,
            dataset.n_train,
            dataset.n_test,
            dataset.input_shape,
            dataset.n_classes,
            delta,
        )
        device = resolve_device(config.device)

        rows: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        for job in jobs:
            status.update(current=job.run_id, updated=_now())
            storage.write_json(run_dir / storage.STATUS, status)
            row, job_history = run_job(config, job, dataset, delta, device, log)
            rows.append(row)
            history.extend(job_history)
            storage.write_csv(run_dir / storage.RESULTS, pd.DataFrame(rows))
            if history:
                storage.write_csv(run_dir / storage.HISTORY, pd.DataFrame(history))
            status["done"] += 1
            status["failed"] += int(row["status"] != "ok")
            log.info(_progress_line(job, len(jobs), row))
            if progress is not None:
                progress(dict(status))

        results = pd.DataFrame(rows)
        summary = summarize(results, [m.label for m in config.mechanisms])
        storage.write_csv(run_dir / storage.SUMMARY, summary)
        log.info("\n%s\n", summary_table(summary))
        status.update(state="finished", current=None, updated=_now(), duration_s=time.perf_counter() - clock)
        storage.write_json(run_dir / storage.STATUS, status)
        log.info("Results: %s", run_dir)
        return ExperimentResult(run_dir, results, summary, pd.DataFrame(history))
    except BaseException as error:
        state = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        status.update(state=state, error=f"{type(error).__name__}: {error}", updated=_now())
        storage.write_json(run_dir / storage.STATUS, status)
        if state == "failed":
            log.exception("The experiment failed")
        raise
    finally:
        _detach_handlers(handlers)
