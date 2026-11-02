"""Registries: registration by decorator, lookup by name, helpful errors."""

import pytest

from dpbench.core.registry import QUERIES, QUERY_MECHANISMS, TABLES, Registry, UnknownPluginError


def test_builtin_plugins_are_registered():
    assert set(TABLES.names()) == {"census_synthetic", "diabetes"}
    assert set(QUERIES.names()) == {
        "count", "sum", "mean", "histogram", "marginal", "centroid", "range_queries", "quantile", "mode",
    }  # fmt: skip
    assert set(QUERY_MECHANISMS.names()) == {"laplace", "gaussian", "polar_laplace", "exponential"}


def test_register_and_get():
    registry = Registry("thing")

    @registry.register("foo")
    class Foo:
        pass

    assert registry.get("foo") is Foo
    assert Foo.name == "foo"  # the decorator gives the class its registered name
    assert "foo" in registry and list(registry) == ["foo"] and len(registry) == 1


def test_duplicate_names_are_rejected():
    registry = Registry("thing")
    registry.register("foo")(type("A", (), {}))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("foo")(type("B", (), {}))


def test_names_must_be_identifiers():
    with pytest.raises(ValueError):
        Registry("thing").register("not valid")


def test_unknown_name_suggests_a_close_match():
    with pytest.raises(UnknownPluginError, match="Did you mean 'laplace'"):
        QUERY_MECHANISMS.get("laplce")
