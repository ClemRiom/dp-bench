"""The workspace: datasets, Python plugins and external programs added without touching the code."""

import math
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from dpbench import workspace
from dpbench.classic.external import LANGUAGES, ExternalMechanismError, check_program, run_program
from dpbench.classic.tasks import NumericTask
from dpbench.core.registry import DATASETS, MECHANISMS, QUERY_MECHANISMS, TABLES, plugins_of_file
from dpbench.core.schema import infer_schema
from dpbench.datasets.uploaded import encode_features
from dpbench.plugin_check import check_plugin_code
from dpbench.templates import (
    ACCOUNTANT_TEMPLATE,
    DATASET_TEMPLATE,
    FUNCTION_TEMPLATE,
    QUERY_TEMPLATE,
    TRAINING_TEMPLATE,
)


@pytest.fixture
def space(tmp_path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV, str(tmp_path / "workspace"))
    return tmp_path / "workspace"


def patients(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "bmi": rng.normal(26, 4, n).round(1),
            "smoker": rng.choice(["yes", "no"], n),
            "city": rng.choice(["Paris", "Lyon", "Nice"], n),
        }
    )
    frame["sick"] = np.where(rng.random(n) < 0.2 + 0.4 * (frame["smoker"] == "yes"), "yes", "no")
    return frame


# --------------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------------


def test_dataset_with_a_target_is_a_table_and_a_dataset(space):
    frame = patients()
    schema = infer_schema(frame)
    schema.target, schema.sensitive_attribute = "sick", "smoker"
    meta = workspace.save_dataset("ws_patients", frame, schema, "toy patients")
    try:
        assert meta["registered_as"] == ["table", "dataset"]
        assert (space / "datasets" / "ws_patients.csv").is_file()
        table = TABLES.get("ws_patients")().load()
        assert table.n == 400 and "city" in table.categorical_columns()
        data = DATASETS.get("ws_patients")(test_size=0.25).load()
        assert data.n_classes == 2 and data.n_train == 300
        assert data.n_features == 2 + 2 + 3  # numeric scaled, one-hot smoker and city (the target excluded)
        assert data.groups_test is not None  # the sensitive attribute, for the fairness metrics
        assert [m["name"] for m in workspace.list_datasets()] == ["ws_patients"]
    finally:
        workspace.delete_dataset("ws_patients")
    assert "ws_patients" not in TABLES and "ws_patients" not in DATASETS
    assert not (space / "datasets" / "ws_patients.csv").exists()


def test_dataset_without_a_target_is_only_a_table(space):
    workspace.save_dataset("ws_counts", patients(), infer_schema(patients()))
    try:
        assert "ws_counts" in TABLES and "ws_counts" not in DATASETS
        with pytest.raises(FileExistsError):
            workspace.save_dataset("ws_counts", patients(), infer_schema(patients()))
        workspace.save_dataset("ws_counts", patients(n=100), infer_schema(patients()), overwrite=True)
        assert TABLES.get("ws_counts")().load().n == 100
    finally:
        workspace.delete_dataset("ws_counts")


def test_dataset_names_are_checked(space):
    with pytest.raises(ValueError, match="built-in"):
        workspace.save_dataset("adult", patients(), infer_schema(patients()))
    with pytest.raises(ValueError, match="not a valid name"):
        workspace.save_dataset("my data", patients(), infer_schema(patients()))


def test_encode_features_uses_the_public_domains():
    frame = pd.DataFrame({"age": [0.0, 50.0, 200.0, np.nan], "city": ["a", "b", "zzz", "a"]})
    schema = infer_schema(frame)
    schema.columns["age"].lower, schema.columns["age"].upper = 0.0, 100.0
    schema.columns["city"].categories = ["a", "b"]
    features, names = encode_features(frame, schema, exclude=())
    assert features[:, 0].tolist() == [0.0, 0.5, 1.0, 0.5]  # clipped to [0, 100], missing -> middle
    assert features[2, 1:].tolist() == [0.0, 0.0]  # an unknown category has no one-hot bit
    assert len(names) == features.shape[1] == 3


# --------------------------------------------------------------------------------------------
# Python plugins
# --------------------------------------------------------------------------------------------


def test_save_reload_and_delete_a_plugin(space):
    result = workspace.save_plugin("ws_lap.py", QUERY_TEMPLATE)
    path = space / "plugins" / "ws_lap.py"
    try:
        assert result["plugins"] == ["query_mechanism:laplace_postprocessed"]
        mechanism = QUERY_MECHANISMS.get("laplace_postprocessed")()
        noisy = mechanism.release(NumericTask([10.0], 1.0, 1.0, integer=True), 1.0, 0.0, np.random.default_rng(0))
        assert float(noisy[0]) == round(float(noisy[0])) and float(noisy[0]) >= 0
        # Saving a new version under the same file name replaces the old plugin.
        renamed = QUERY_TEMPLATE.replace('"laplace_postprocessed"', '"laplace_rounded"')
        workspace.save_plugin("ws_lap.py", renamed)
        assert "laplace_rounded" in QUERY_MECHANISMS and "laplace_postprocessed" not in QUERY_MECHANISMS
        assert plugins_of_file(path) == [("query_mechanism", "laplace_rounded")]
    finally:
        workspace.delete_plugin("ws_lap.py")
    assert "laplace_rounded" not in QUERY_MECHANISMS and not path.exists()


def test_a_broken_plugin_is_not_saved_and_the_old_version_survives(space):
    workspace.save_plugin("ws_keep.py", QUERY_TEMPLATE)
    try:
        with pytest.raises(SyntaxError):
            workspace.save_plugin("ws_keep.py", "def broken(:\n")
        assert "laplace_postprocessed" in QUERY_MECHANISMS
        assert "register_query_mechanism" in (space / "plugins" / "ws_keep.py").read_text(encoding="utf-8")
        with pytest.raises(SyntaxError):
            workspace.save_plugin("ws_new.py", "def broken(:\n")
        assert not (space / "plugins" / "ws_new.py").exists()
    finally:
        workspace.delete_plugin("ws_keep.py")


def test_load_workspace_reports_errors_without_raising(space):
    (space / "plugins").mkdir(parents=True)
    (space / "plugins" / "ws_bad.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    errors = workspace.load_workspace()
    assert "ws_bad.py" in errors and "boom" in errors["ws_bad.py"]
    assert workspace.list_plugins()[0]["error"]


def test_training_template_registers_a_mechanism(space):
    workspace.save_plugin("ws_autoclip.py", TRAINING_TEMPLATE)
    try:
        assert "dpsgd_autoclip" in MECHANISMS
        assert MECHANISMS.get("dpsgd_autoclip").title
    finally:
        workspace.delete_plugin("ws_autoclip.py")
    assert "dpsgd_autoclip" not in MECHANISMS


# --------------------------------------------------------------------------------------------
# Checking a plugin in a separate process
# --------------------------------------------------------------------------------------------


def test_check_plugin_code_accepts_the_templates():
    report = check_plugin_code(QUERY_TEMPLATE, "laplace_postprocessed.py")
    assert report["ok"], report
    assert report["plugins"][0]["kind"] == "query_mechanism"
    report = check_plugin_code(DATASET_TEMPLATE, "wine_plugin.py")
    assert report["ok"], report
    report = check_plugin_code(FUNCTION_TEMPLATE, "my_laplace.py")
    assert report["ok"], report
    assert report["plugins"][0]["name"] == "my_laplace"
    assert report["plugins"][0]["noise_std"] == pytest.approx(math.sqrt(2), rel=0.2)  # Lap(1/ε) with ε = 1
    report = check_plugin_code(ACCOUNTANT_TEMPLATE, "no_amplification.py")
    assert report["ok"], report
    assert report["plugins"][0]["epsilon"] > 100  # without amplification by subsampling, 1000 steps cost a lot


@pytest.mark.slow
def test_check_plugin_code_trains_a_training_algorithm():
    report = check_plugin_code(TRAINING_TEMPLATE, "dpsgd_autoclip.py")
    assert report["ok"], report
    assert 0 < report["plugins"][0]["epsilon_spent"] <= 2.0 + 1e-9


@pytest.mark.parametrize(
    "code, message",
    [
        ("def f(:\n    pass\n", "SyntaxError"),
        ("x = 1\n", "defines no plugin"),
        ("import not_a_module_xyz\n", "not_a_module_xyz"),
        (
            QUERY_TEMPLATE.replace("        return noisy", "        return 'oops'"),
            "laplace_postprocessed",
        ),
    ],
    ids=["syntax", "nothing", "import", "bad release"],
)
def test_check_plugin_code_reports_problems(code, message):
    report = check_plugin_code(code, "candidate.py")
    assert not report["ok"]
    assert message in str(report)


# --------------------------------------------------------------------------------------------
# Mechanisms written in another language
# --------------------------------------------------------------------------------------------


def program(tmp_path, language, code=None):
    path = tmp_path / f"program{LANGUAGES[language].extension}"
    path.write_text(code if code is not None else LANGUAGES[language].template, encoding="utf-8")
    return path


@pytest.mark.parametrize("language", ["python", "javascript", "bash", "java", "r"])
def test_language_templates_follow_the_protocol(tmp_path, language):
    if not LANGUAGES[language].available:
        pytest.skip(f"{LANGUAGES[language].command[0]} is not installed")
    report = check_program(language, program(tmp_path, language), n_values=3000)
    assert report["ok"], report
    assert report["random"] and report["reproducible"]
    assert report["noise_std"] == pytest.approx(report["laplace_std"], rel=0.12)


def test_protocol_errors(tmp_path):
    command = [sys.executable, str(program(tmp_path, "python", "import sys\nsys.exit(3)\n"))]
    with pytest.raises(ExternalMechanismError, match="exited with code 3"):
        run_program(command, np.array([1.0]), 1.0, 0.0, 1.0, 1.0, seed=0)
    command = [sys.executable, str(program(tmp_path, "python", "print(1)\nprint(2)\n"))]
    with pytest.raises(ExternalMechanismError, match="2 numbers instead of 1"):
        run_program(command, np.array([1.0]), 1.0, 0.0, 1.0, 1.0, seed=0)
    command = [sys.executable, str(program(tmp_path, "python", "print('x')\n"))]
    with pytest.raises(ExternalMechanismError, match="one number per line"):
        run_program(command, np.array([1.0]), 1.0, 0.0, 1.0, 1.0, seed=0)


def test_external_mechanism_in_the_workspace(space):
    manifest = workspace.save_external("ws_lap_py", "python", LANGUAGES["python"].template, "Laplace in Python")
    try:
        assert manifest["program"] == "ws_lap_py.py"
        mechanism = QUERY_MECHANISMS.get("ws_lap_py")()
        task = NumericTask(np.full(5, 100.0), 1.0, 1.0)
        noisy = mechanism.release(task, 1.0, 0.0, np.random.default_rng(0))
        assert noisy.shape == (5,) and np.all(np.abs(noisy - 100) < 30)
        assert [m["name"] for m in workspace.list_external()] == ["ws_lap_py"]
        with pytest.raises(ValueError, match="already the name"):
            workspace.save_external("laplace", "python", LANGUAGES["python"].template)
    finally:
        workspace.delete_external("ws_lap_py")
    assert "ws_lap_py" not in QUERY_MECHANISMS
    assert not list((space / "external").iterdir())


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_external_mechanism_in_a_benchmark(space):
    from dpbench.classic.runner import ClassicConfig, Spec, run_query_benchmark

    workspace.save_external("ws_lap_js", "javascript", LANGUAGES["javascript"].template)
    try:
        config = ClassicConfig(
            name="external",
            tables=[Spec("diabetes")],
            queries=[Spec("count"), Spec("histogram", {"column": "sex"})],
            mechanisms=[Spec("laplace"), Spec("ws_lap_js")],
            epsilons=[1.0],
            repetitions=3,
        )
        summary = run_query_benchmark(config).summary
        assert set(summary["mechanism"]) == {"laplace", "ws_lap_js"}
        assert (summary["rel_error_mean"] < 0.05).all()
    finally:
        workspace.delete_external("ws_lap_js")
