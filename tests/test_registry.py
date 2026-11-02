import textwrap

import pytest

from dpbench.core.registry import (
    ACCOUNTANTS,
    DATASETS,
    MECHANISMS,
    METRICS,
    MODELS,
    Registry,
    UnknownPluginError,
    load_plugins,
)


def test_builtin_plugins_are_registered():
    assert {"non_private", "dpsgd", "dpsgd_opacus", "dp_adam", "dp_lap_sgd"} <= set(MECHANISMS.names())
    assert {"synthetic", "digits", "adult", "mnist", "fashion_mnist", "cifar10"} <= set(DATASETS.names())
    assert {"mlp", "cnn", "cnn_gn"} <= set(MODELS.names())
    assert {"basic", "advanced", "rdp", "prv", "pld", "gdp"} <= set(ACCOUNTANTS.names())
    assert {"accuracy", "f1_macro", "roc_auc", "loss", "fairness_gap"} <= set(METRICS.names())


def test_register_get_and_describe():
    registry = Registry("thing")

    @registry.register("foo")
    class Foo:
        """A foo.

        More details."""

        def __init__(self, size: int = 3, label: str = "x"):
            pass

    assert registry.get("foo") is Foo
    assert Foo.name == "foo"
    info = registry.describe("foo")
    assert info["summary"] == "A foo."
    assert info["parameters"] == {"size": 3, "label": "x"}


def test_duplicate_names_are_rejected():
    registry = Registry("thing")
    registry.register("foo")(type("A", (), {}))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("foo")(type("B", (), {}))


def test_names_must_be_identifiers():
    with pytest.raises(ValueError):
        Registry("thing").register("not valid")


def test_unknown_name_suggests_a_close_match():
    with pytest.raises(UnknownPluginError, match="Did you mean 'dpsgd'"):
        MECHANISMS.get("dpsdg")


def test_unknown_parameter_is_reported():
    with pytest.raises(ValueError, match="Did you mean 'max_grad_norm'"):
        MECHANISMS.check_params("dpsgd", {"max_grad_nrom": 1.0})
    MECHANISMS.check_params("dpsgd", {"max_grad_norm": 1.0, "lr": 0.1})


def test_model_parameters_exclude_shape_arguments():
    assert "input_shape" not in MODELS.parameters("mlp")
    with pytest.raises(ValueError):
        MODELS.check_params("mlp", {"n_classes": 3})


def test_external_plugin_file(tmp_path):
    path = tmp_path / "my_metric.py"
    path.write_text(
        textwrap.dedent('''
        from dpbench.core.registry import register_metric
        from dpbench.metrics import Metric

        @register_metric("always_one")
        class AlwaysOne(Metric):
            """Always 1."""

            def __call__(self, y_true, proba, groups=None):
                return 1.0
    ''')
    )
    try:
        load_plugins([str(path)])
        assert "always_one" in METRICS
        load_plugins([str(path)])  # loading twice is harmless
        assert METRICS.get("always_one")()([0], [[1.0]]) == 1.0
    finally:
        METRICS.unregister("always_one")


def test_missing_plugin_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_plugins([str(tmp_path / "missing.py")])
