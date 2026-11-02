"""Colours, labels and display names shared by the figures and the dashboard."""

from __future__ import annotations

from collections.abc import Iterable

from dpbench.core.registry import MECHANISMS, METRICS

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#7b4fd6", "#e34948", "#c99000", "#e87ba4", "#008300", "#0f8fa8"]
BASELINE_COLOR = "#3d3d3d"
BOUND_COLOR = "#111111"
GRID_COLOR = "#ececec"
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

COLUMN_LABELS = {
    "test_accuracy": "Test accuracy",
    "test_f1_macro": "Test macro F1",
    "test_roc_auc": "Test ROC AUC",
    "test_loss": "Test cross-entropy",
    "test_fairness_gap": "Accuracy gap between groups",
    "test_demographic_parity_gap": "Demographic parity gap",
    "test_class_accuracy_gap": "Accuracy gap between classes",
    "train_accuracy": "Training accuracy",
    "generalization_gap": "Generalization gap (train − test accuracy)",
    "epsilon_target": "Target ε",
    "epsilon_spent": "ε spent",
    "epsilon_opacus": "ε (Opacus' accountant)",
    "epsilon_worst_case": "ε certified (worst case)",
    "epsilon_paper_claim": "ε claimed by the paper",
    "noise_std_radius": "Radial noise std",
    "noise_std_angle": "Angular noise std",
    "noise_multiplier": "Noise multiplier σ",
    "sample_rate": "Sampling rate q",
    "steps": "Steps",
    "train_time_s": "Training time (s)",
    "time_per_epoch_s": "Time per epoch (s)",
    "calibration_time_s": "Noise calibration time (s)",
    "eval_time_s": "Evaluation time (s)",
    "peak_gpu_memory_mb": "Peak GPU memory (MB)",
    "n_parameters": "Parameters",
}

UTILITY_COLUMNS = ["test_accuracy", "test_f1_macro", "test_roc_auc", "test_loss"]
FAIRNESS_COLUMNS = ["test_fairness_gap", "test_demographic_parity_gap", "test_class_accuracy_gap"]


def column_label(column: str) -> str:
    if column in COLUMN_LABELS:
        return COLUMN_LABELS[column]
    if column.startswith("test_") and column[5:] in METRICS:
        label = METRICS.get(column[5:]).label
        if label:
            return f"Test {label[0].lower()}{label[1:]}"
    return column.replace("_", " ").capitalize()


def lower_is_better(column: str) -> bool:
    if column.startswith("test_") and column[5:] in METRICS:
        return not METRICS.get(column[5:]).higher_is_better
    return column in {
        "generalization_gap",
        "train_time_s",
        "time_per_epoch_s",
        "epsilon_spent",
    }


def display_name(label: str, mechanism_name: str | None = None) -> str:
    """Pretty name of a mechanism: its ``title`` when the configuration kept the default label."""
    name = mechanism_name or label
    if label == name and name in MECHANISMS:
        title = getattr(MECHANISMS.get(name), "title", "")
        if title:
            return title
    return label


def color_map(labels: Iterable[str], baselines: Iterable[str] = ()) -> dict[str, str]:
    """A stable colour per mechanism label; the non-private baselines are dark grey."""
    baseline_set = set(baselines)
    colors, index = {}, 0
    for label in labels:
        if label in colors:
            continue
        if label in baseline_set:
            colors[label] = BASELINE_COLOR
        else:
            colors[label] = PALETTE[index % len(PALETTE)]
            index += 1
    return colors


#: Short names of the built-in accountants (an accountant added as a plugin keeps its registry name).
ACCOUNTANT_NAMES = {
    "basic": "Basic composition",
    "advanced": "Advanced composition",
    "rdp": "RDP",
    "prv": "PRV",
    "pld": "PLD",
    "gdp": "GDP",
}


def accountant_order(names: Iterable[str]) -> list[str]:
    """The built-in accountants from the loosest to the tightest, then the others (plugins) by name."""
    names = list(names)
    return [n for n in ACCOUNTANT_NAMES if n in names] + sorted(n for n in names if n not in ACCOUNTANT_NAMES)


def accountant_name(name: str) -> str:
    return ACCOUNTANT_NAMES.get(name, name)


def eps_text(epsilon: float) -> str:
    return "∞" if epsilon == float("inf") else f"{epsilon:g}"
