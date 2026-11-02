"""Run page: build an experiment with the form, launch it in the background and follow it.

The form becomes a configuration (``ExperimentConfig``), validated before it can be run.
The experiment runs in a separate process (``dpbench run``), which writes its results as
it goes; the Results page reads them.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from dpbench import workspace
from dpbench.core.config import ConfigError, ExperimentConfig, load_config_text
from dpbench.core.registry import ACCOUNTANTS, DATASETS, MECHANISMS, METRICS, MODELS
from dpbench.dashboard import common
from dpbench.experiments.runner import plan_jobs
from dpbench.reporting.style import accountant_name, display_name

EPSILON_CHOICES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def _validate(text: str, base_dir: Path | None) -> tuple[ExperimentConfig | None, str | None]:
    try:
        return load_config_text(text, base_dir=base_dir), None
    except (ConfigError, FileNotFoundError, ValueError) as error:
        return None, str(error)


def _absolute_plugins(config: ExperimentConfig) -> dict[str, Any]:
    data = config.to_dict()
    plugins = []
    for spec in config.plugins:
        candidate = Path(config.base_dir or ".") / spec
        plugins.append(str(candidate.resolve()) if spec.endswith(".py") and candidate.is_file() else spec)
    data["plugins"] = plugins
    return data


def _launch(config: ExperimentConfig) -> None:
    root = common.run_roots()[0]
    staging = root / ".dashboard"
    staging.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{config.slug}"
    config_path = staging / f"{stamp}_{config.slug}.yaml"
    log_path = staging / f"{stamp}_{config.slug}.log"
    config_path.write_text(
        yaml.safe_dump(_absolute_plugins(config), sort_keys=False, allow_unicode=True, width=1000), encoding="utf-8"
    )
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "dpbench.cli", "run", str(config_path), "--run-dir", str(run_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env={**os.environ, workspace.WORKSPACE_ENV: str(workspace.workspace_dir())},
            start_new_session=os.name == "posix",
        )
    st.session_state.setdefault("processes", {})[process.pid] = process
    st.session_state.setdefault("jobs", []).append(
        {
            "pid": process.pid,
            "name": config.name,
            "run_dir": str(run_dir),
            "log": str(log_path),
            "total": len(plan_jobs(config)),
            "launched": time.time(),
        }
    )
    st.toast(f"Experiment {config.name!r} started")


def _plan_summary(config: ExperimentConfig) -> str:
    jobs = plan_jobs(config)
    mechanisms = ", ".join(display_name(m.label, m.name) for m in config.mechanisms)
    return (
        f"{len(jobs)} trainings: {mechanisms}; ε ∈ {config.privacy.epsilons}; "
        f"{len(config.seeds)} seed(s); {config.training.epochs} epochs."
    )


COMMON_PARAMETERS = {
    "lr",
    "max_grad_norm",
    "optimizer",
    "momentum",
    "weight_decay",
    "accountant",
    "noise_multiplier",
    "max_physical_batch_size",
}


def _extra_parameters(mechanism: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Inputs for the parameters specific to a mechanism (e.g. beta and accounting for GeoDP)."""
    choices = getattr(MECHANISMS.get(mechanism), "parameter_choices", {})
    extra = [k for k, v in defaults.items() if k not in COMMON_PARAMETERS and isinstance(v, (bool, int, float, str))]
    params: dict[str, Any] = {}
    for key, column in zip(extra, itertools.cycle(st.columns(3))):
        value, widget_key = defaults[key], f"{key}_{mechanism}"
        if key in choices:
            options = list(choices[key])
            params[key] = column.selectbox(
                key, options, index=options.index(value) if value in options else 0, key=widget_key
            )
        elif isinstance(value, bool):
            params[key] = column.checkbox(key, value=value, key=widget_key)
        elif isinstance(value, (int, float)):
            params[key] = column.number_input(key, value=float(value), format="%.4g", key=widget_key)
            if isinstance(value, int):
                params[key] = int(params[key])
        else:
            params[key] = column.text_input(key, value=value, key=widget_key)
    return params


def _with_accountants(specs: list[dict[str, Any]], accountants: list[str]) -> list[dict[str, Any]]:
    """One training per (private mechanism, chosen accountant that can analyse its noise).

    With one accountant, the labels stay the mechanisms' names; with several, each copy is
    labelled with its accountant, e.g. "DP-SGD (RDP)". A mechanism that none of the chosen
    accountants can analyse (Laplace noise and RDP, for instance) keeps its own default.
    """
    expanded = []
    for spec in specs:
        cls = MECHANISMS.get(spec["name"])
        noise = getattr(cls, "noise", None) or "gaussian"
        usable = [a for a in accountants if noise in ACCOUNTANTS.get(a).supported_noises]
        if not cls.private or not usable:
            expanded.append(spec)
            continue
        for accountant in usable:
            params = {**spec["params"], "accountant": accountant}
            label = f"{display_name(spec['name'])} ({accountant_name(accountant)})" if len(accountants) > 1 else ""
            expanded.append(
                {"name": spec["name"], "label": label, "params": params}
                if label
                else {"name": spec["name"], "params": params}
            )
    return expanded


def _form() -> None:
    datasets, models = DATASETS.names(), MODELS.names()
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Experiment name", "dashboard-experiment")
    dataset = c2.selectbox("Dataset", datasets, index=datasets.index("digits") if "digits" in datasets else 0)
    model = c3.selectbox("Model", models, index=models.index("mlp") if "mlp" in models else 0)
    c1, c2, c3 = st.columns(3)
    n_train = c1.number_input("Training examples (0 = all)", min_value=0, value=0, step=1000)
    epochs = c2.number_input("Epochs", min_value=1, max_value=1000, value=20)
    batch_size = c3.number_input("Batch size", min_value=4, max_value=16384, value=64, step=32)
    c1, c2, c3 = st.columns(3)
    seeds = c1.number_input("Number of seeds", min_value=1, max_value=20, value=2)
    delta = c2.text_input("δ (a number, or auto for 1/N)", "1e-5")
    hidden = c3.text_input("Hidden layers (mlp only)", "64", disabled=model != "mlp")
    epsilons = st.multiselect("Privacy budgets ε", EPSILON_CHOICES, default=[1.0, 4.0, 16.0])
    mechanisms = st.multiselect(
        "Mechanisms",
        MECHANISMS.names(),
        default=[m for m in ("non_private", "dpsgd", "dp_lap_sgd") if m in MECHANISMS],
        format_func=display_name,
    )
    accountants = st.multiselect(
        "Accountants (several to compare them)",
        ACCOUNTANTS.names(),
        default=[a for a in ("prv",) if a in ACCOUNTANTS],
        format_func=accountant_name,
    )
    specs = []
    for mechanism in mechanisms:
        defaults = MECHANISMS.parameters(mechanism)
        with st.expander(f"Hyper-parameters of {display_name(mechanism)}"):
            c1, c2, c3 = st.columns(3)
            params: dict[str, Any] = {
                "lr": c1.number_input(
                    "Learning rate",
                    min_value=1e-6,
                    value=float(defaults.get("lr", 0.1)),
                    format="%.4g",
                    key=f"lr_{mechanism}",
                )
            }
            if "max_grad_norm" in defaults:
                params["max_grad_norm"] = c2.number_input(
                    "Clipping bound C",
                    min_value=1e-4,
                    value=float(defaults["max_grad_norm"]),
                    format="%.4g",
                    key=f"clip_{mechanism}",
                )
            if "optimizer" in defaults:
                options = ["sgd", "adam", "adamw"]
                params["optimizer"] = c3.selectbox(
                    "Optimizer", options, index=options.index(defaults["optimizer"]), key=f"opt_{mechanism}"
                )
            params.update(_extra_parameters(mechanism, defaults))
            specs.append({"name": mechanism, "params": params})
    metrics = st.multiselect(
        "Metrics",
        METRICS.names(),
        default=[
            m for m in ("accuracy", "f1_macro", "roc_auc", "loss", "fairness_gap", "class_accuracy_gap") if m in METRICS
        ],
    )
    try:
        delta_value: Any = "auto" if delta.strip() == "auto" else float(delta)
    except ValueError:
        delta_value = delta
    dataset_params: dict[str, Any] = {"n_train": int(n_train)} if n_train else {}
    model_params: dict[str, Any] = {}
    if model == "mlp":
        try:
            model_params["hidden"] = [int(h) for h in hidden.replace(",", " ").split()]
        except ValueError:
            st.error("Hidden layers must be integers separated by spaces or commas.")
    data = {
        "name": name,
        "description": "Built with the dashboard form.",
        "seeds": list(range(int(seeds))),
        "dataset": {"name": dataset, "params": dataset_params},
        "model": {"name": model, "params": model_params},
        "training": {"epochs": int(epochs), "batch_size": int(batch_size)},
        "privacy": {"epsilons": epsilons, "delta": delta_value},
        "mechanisms": _with_accountants(specs, accountants),
        "metrics": metrics,
    }
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000)
    config, error = _validate(text, None)
    if error:
        st.error(error)
    else:
        st.success(_plan_summary(config))
    if st.button("Run", type="primary", disabled=config is None, key="run_form"):
        _launch(config)


def _job_state(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    status = common.read_status(job["run_dir"])
    process = st.session_state.get("processes", {}).get(job["pid"])
    code = process.poll() if process is not None else None
    state = status.get("state", "starting")
    if code is not None and state in ("starting", "running"):
        state = "failed" if code != 0 else "finished"
    return state, status


@st.fragment(run_every=2)
def _jobs() -> None:
    jobs = st.session_state.get("jobs", [])
    if not jobs:
        st.write("No experiment started yet.")
        return
    for job in reversed(jobs):
        state, status = _job_state(job)
        total = int(status.get("total") or job["total"] or 1)
        done = int(status.get("done") or 0)
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(
                f"**{job['name']}**, {state}: {done}/{total} trainings"
                + (f", now `{status['current']}`" if state == "running" and status.get("current") else "")
            )
            left.progress(min(done / total, 1.0))
            if state in ("finished", "running") and Path(job["run_dir"], "results.csv").exists():
                if right.button("Open results", key=f"open_{job['pid']}"):
                    common.select_run(str(Path(job["run_dir"]).resolve()))
                    if "results" in common.PAGES:
                        st.switch_page(common.PAGES["results"])
            if state == "running":
                process = st.session_state.get("processes", {}).get(job["pid"])
                if process is not None and right.button("Stop", key=f"stop_{job['pid']}"):
                    process.terminate()
            log = Path(job["log"])
            if log.is_file():
                with st.expander("Log"):
                    st.code(
                        "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]) or "(empty)",
                        language=None,
                    )


def render() -> None:
    st.title("Run an experiment")
    _form()
    st.subheader("Experiments started here")
    _jobs()
