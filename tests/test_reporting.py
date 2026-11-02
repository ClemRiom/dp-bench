import plotly.graph_objects as go
import pytest

from dpbench.experiments.storage import load_run
from dpbench.reporting import figures
from dpbench.reporting.figures import nice_log_ticks
from dpbench.reporting.tables import calibration_table, key_findings, leaderboard, summary_table


@pytest.fixture(scope="module")
def run(tiny_run):
    return load_run(tiny_run)


def test_figures_build(run):
    summary = run.summary()
    mechanism = run.private_mechanisms[0]
    built = [
        figures.privacy_utility(summary, "test_accuracy", ("Majority class", 0.5)),
        figures.noise_vs_epsilon(summary),
        figures.epsilon_check(summary),
        figures.epsilon_over_training(run.history, mechanism),
    ]
    for fig in built:
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


def test_tables(run):
    summary = run.summary()
    board, best = leaderboard(summary, "test_accuracy")
    assert list(board.index) == ["∞", "1", "4"]
    assert best.loc["1"].sum() == 1 and not best.loc["∞"].any()
    assert len(summary_table(summary, ["test_accuracy", "epsilon_spent"])) == 3
    assert len(calibration_table(summary)) == 2
    findings = key_findings(summary, run.results)
    assert any("Accounting check" in f for f in findings)


def test_nice_log_ticks():
    assert nice_log_ticks(1, 16) == [1, 2, 5, 10, 20]
    assert nice_log_ticks(1, 8) == [1, 2, 3, 5, 10]
    assert nice_log_ticks(0.001, 1) == [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    assert nice_log_ticks(-1, 2) == []
