"""Accountants page: the epsilon of a DP-SGD training according to every accountant, and the noise each one needs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dpbench.accounting.calibration import CalibrationError, calibrate_noise
from dpbench.core.registry import ACCOUNTANTS
from dpbench.dashboard import common
from dpbench.mechanisms.base import SamplingPlan
from dpbench.reporting import figures
from dpbench.reporting.style import accountant_name, accountant_order


@st.cache_data(show_spinner=False, max_entries=1024)
def _epsilon(name: str, sigma: float, rate: float, steps: int, delta: float, noise: str) -> float:
    return ACCOUNTANTS.get(name)().epsilon(sigma, rate, steps, delta, noise)


@st.cache_data(show_spinner=False, max_entries=256)
def _sigma(name: str, target: float, rate: float, steps: int, delta: float, noise: str) -> float | None:
    try:
        return calibrate_noise(ACCOUNTANTS.get(name)(), target, rate, steps, delta, noise)
    except CalibrationError:
        return None


def _kind(name: str) -> str:
    return "approximation" if ACCOUNTANTS.get(name).approximate else "upper bound"


def render() -> None:
    st.title("Privacy accountants")
    c = st.columns(6)
    n = c[0].number_input("Dataset size N", min_value=1, value=60_000, step=1000)
    batch = c[1].number_input("Batch size", min_value=1, value=256, step=32)
    epochs = c[2].number_input("Epochs", min_value=1, value=10)
    sigma = c[3].number_input("Noise multiplier σ", min_value=0.3, value=1.0, step=0.1, format="%.3f")
    delta = c[4].number_input("δ", min_value=1e-12, max_value=0.5, value=1e-5, format="%.1e")
    noise = c[5].selectbox("Noise", ["gaussian", "laplace"])
    plan = SamplingPlan(int(n), int(batch), int(epochs))
    left, middle, right = st.columns(3)
    left.metric("Sampling rate q", f"{plan.sample_rate:.4g}")
    middle.metric("Steps per epoch", f"{plan.steps_per_epoch:,}")
    right.metric("Steps T", f"{plan.steps:,}")

    names = accountant_order(n for n in ACCOUNTANTS.names() if noise in ACCOUNTANTS.get(n).supported_noises)
    st.subheader(f"ε after {plan.steps:,} steps with σ = {sigma:g}")
    rows = []
    with st.spinner("Computing ε..."):
        for name in names:
            value = _epsilon(name, float(sigma), plan.sample_rate, plan.steps, float(delta), noise)
            rows.append({"Accountant": accountant_name(name), "ε": round(value, 4), "Result": _kind(name)})
    common.table(pd.DataFrame(rows))

    st.subheader("ε as the training goes on")
    points = sorted({max(1, int(round(e))) for e in np.linspace(1, int(epochs), min(int(epochs), 8))})
    curve = []
    with st.spinner("Computing the curves..."):
        for name in names:
            for epoch in points:
                value = _epsilon(
                    name, float(sigma), plan.sample_rate, epoch * plan.steps_per_epoch, float(delta), noise
                )
                curve.append(
                    {
                        "accountant": name,
                        "epoch": epoch,
                        "epsilon": value,
                        "approximate": ACCOUNTANTS.get(name).approximate,
                    }
                )
    common.plot(figures.accountant_curves(pd.DataFrame(curve)), key="fig_accountants")

    st.subheader("Noise needed for a target ε")
    target = st.number_input("Target ε", min_value=0.01, value=1.0, step=0.5)
    sigmas = []
    with st.spinner("Calibrating..."):
        for name in names:
            value = _sigma(name, float(target), plan.sample_rate, plan.steps, float(delta), noise)
            sigmas.append(
                {
                    "Accountant": accountant_name(name),
                    "Noise multiplier σ": "out of reach" if value is None else round(value, 4),
                    "Result": _kind(name),
                }
            )
    common.table(pd.DataFrame(sigmas))
