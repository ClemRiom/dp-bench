import pytest
import yaml

from conftest import tiny_config
from dpbench.cli import main

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def results_page():
    from dpbench.dashboard.views import results

    results.render()


def run_page():
    from dpbench.dashboard.views import run

    run.render()


def classic_page():
    from dpbench.dashboard.views import classic

    classic.render()


def accountants_page():
    from dpbench.dashboard.views import accountants

    accountants.render()


def add_dataset_page():
    from dpbench.dashboard.views import add_dataset

    add_dataset.render()


def add_algorithm_page():
    from dpbench.dashboard.views import add_algorithm

    add_algorithm.render()


PAGES = [run_page, results_page, accountants_page, classic_page, add_dataset_page, add_algorithm_page]


@pytest.mark.parametrize("page", PAGES, ids=lambda f: f.__name__)
def test_dashboard_pages_render(page, tiny_run, monkeypatch, tmp_path):
    monkeypatch.setenv("DPBENCH_RUNS_DIRS", str(tiny_run.parent))
    monkeypatch.setenv("DPBENCH_WORKSPACE", str(tmp_path / "workspace"))
    app = streamlit_testing.AppTest.from_function(page, default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_classic_page_runs_a_benchmark(monkeypatch, tmp_path):
    monkeypatch.setenv("DPBENCH_RUNS_DIRS", str(tmp_path / "runs"))
    app = streamlit_testing.AppTest.from_function(classic_page, default_timeout=300)
    app.run()
    app.slider(key="reps_one").set_value(5)
    app.multiselect(key="eps_one").set_value([0.5, 2.0])
    next(b for b in app.button if b.label == "Run the benchmark").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "ε spent = target ε" and m.value == "yes" for m in app.metric)


def test_classic_page_compares_databases(monkeypatch, tmp_path):
    monkeypatch.setenv("DPBENCH_RUNS_DIRS", str(tmp_path / "runs"))
    app = streamlit_testing.AppTest.from_function(classic_page, default_timeout=300)
    app.run()
    app.multiselect(key="compare_mechanisms").set_value(["laplace", "exponential"])
    app.multiselect(key="eps_compare").set_value([1.0])
    app.slider(key="reps_compare").set_value(5)
    next(b for b in app.button if b.label == "Compare the databases").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Databases × queries" and m.value.startswith("3 ×") for m in app.metric)


def test_add_algorithm_page_checks_then_saves(monkeypatch, tmp_path):
    monkeypatch.setenv("DPBENCH_WORKSPACE", str(tmp_path / "workspace"))
    app = streamlit_testing.AppTest.from_function(add_algorithm_page, default_timeout=300)
    app.run()
    app.radio[0].set_value("query")
    app.run()
    save = next(b for b in app.button if b.label == "Save to the workspace")
    assert save.disabled  # not before a successful check
    next(b for b in app.button if b.label == "Check the code").click()
    app.run()
    assert any("test passed" in s.value for s in app.success)
    next(b for b in app.button if b.label == "Save to the workspace").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert (tmp_path / "workspace" / "plugins" / "laplace_postprocessed.py").is_file()
    from dpbench import workspace

    workspace.delete_plugin("laplace_postprocessed.py")


def test_cli_run(capsys, tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(tiny_config()), encoding="utf-8")
    assert main(["run", str(path), "--dry-run", "seeds=[0]"]) == 0
    assert "2 trainings" in capsys.readouterr().out
    assert main(["run", str(path), "trainin.epochs=1"]) == 2
    assert "Did you mean 'training'" in capsys.readouterr().err
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2


def test_one_training_per_mechanism_and_accountant():
    from dpbench.dashboard.views.run import _with_accountants

    specs = [{"name": n, "params": {"lr": 0.1}} for n in ("non_private", "dpsgd", "dp_lap_sgd")]
    one = _with_accountants(specs, ["prv"])
    assert [s.get("label", s["name"]) for s in one] == ["non_private", "dpsgd", "dp_lap_sgd"]
    assert one[1]["params"]["accountant"] == "prv" and "accountant" not in one[2]["params"]  # PRV: Gaussian only
    several = _with_accountants(specs, ["rdp", "pld"])
    labels = [s.get("label", s["name"]) for s in several]
    assert labels == ["non_private", "DP-SGD (RDP)", "DP-SGD (PLD)", "DP-LAP-SGD (PLD)"]
    assert [s["params"].get("accountant") for s in several] == [None, "rdp", "pld", "pld"]
    assert _with_accountants(specs, []) == specs  # no choice: every mechanism keeps its own accountant


def test_run_form_compares_accountants():
    app = streamlit_testing.AppTest.from_function(run_page, default_timeout=120)
    app.run()
    accountants = next(m for m in app.multiselect if m.label.startswith("Accountants"))
    accountants.set_value(["rdp", "prv"]).run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("DP-SGD (RDP)" in s.value and "DP-SGD (PRV)" in s.value for s in app.success)
