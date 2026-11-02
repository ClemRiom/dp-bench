import pandas as pd
import pytest

from conftest import tiny_config
from dpbench.core.config import config_from_dict
from dpbench.core.registry import MECHANISMS, register_mechanism
from dpbench.experiments.runner import run_experiment
from dpbench.experiments.storage import find_runs, load_run
from dpbench.mechanisms.non_private import NonPrivate


def test_end_to_end_run(tiny_run):
    files = {p.name for p in tiny_run.iterdir()}
    assert {
        "config.yaml",
        "environment.json",
        "dataset.json",
        "results.csv",
        "summary.csv",
        "history.csv",
        "run.log",
        "status.json",
    } <= files
    run = load_run(tiny_run)
    assert run.status["state"] == "finished"
    assert len(run.results) == 2 + 2 * 2  # baseline x 2 seeds + dpsgd x 2 budgets x 2 seeds
    assert (run.results["status"] == "ok").all()
    private = run.results[run.results["private"]]
    assert (private["epsilon_spent"] <= private["epsilon_target"]).all()
    for column in (
        "test_accuracy",
        "test_fairness_gap",
        "noise_multiplier",
        "train_time_s",
    ):
        assert column in run.results
    summary = run.summary()
    assert len(summary) == 3 and (summary["n_seeds"] == 2).all()
    assert set(run.history["run_id"]) == set(run.results["run_id"])
    assert find_runs([tiny_run.parent]) == [tiny_run.resolve()]


def test_a_failing_training_does_not_stop_the_experiment(tmp_path):
    @register_mechanism("always_fails")
    class AlwaysFails(NonPrivate):
        """Raises during training."""

        def fit(self, *args, **kwargs):
            raise RuntimeError("boom")

    try:
        config = config_from_dict(tiny_config(mechanisms=["always_fails", "non_private"]))
        result = run_experiment(config, tmp_path, console=False)
    finally:
        MECHANISMS.unregister("always_fails")
    statuses = dict(zip(result.results["mechanism"], result.results["status"]))
    assert statuses == {"always_fails": "failed", "non_private": "ok"}
    assert "boom" in result.results.loc[result.results["mechanism"] == "always_fails", "error"].iloc[0]
    assert "boom" in (result.directory / "run.log").read_text()


def test_explicit_run_directory(tmp_path):
    config = config_from_dict(tiny_config(mechanisms=["non_private"]))
    target = tmp_path / "my-run"
    result = run_experiment(config, run_dir=target, console=False)
    assert result.directory == target
    assert isinstance(pd.read_csv(target / "results.csv"), pd.DataFrame)
    with pytest.raises(FileExistsError):
        run_experiment(config, run_dir=target, console=False)


def test_saved_configuration_keeps_plugin_files_findable(tmp_path, monkeypatch):
    from dpbench.experiments.runner import portable_config

    plugin = tmp_path / "plugins" / "my_metric.py"
    plugin.parent.mkdir()
    plugin.write_text("")
    config = config_from_dict(tiny_config(plugins=["plugins/my_metric.py"]), base_dir=tmp_path)
    monkeypatch.chdir(tmp_path)
    assert "- plugins/my_metric.py" in portable_config(config)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert str(plugin.resolve()) in portable_config(config)  # outside the current directory: absolute path
