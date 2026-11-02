"""Core of DP-Bench: plugin registries, table schemas, seeding."""

from dpbench.core.registry import (
    QUERIES,
    QUERY_MECHANISMS,
    REGISTRIES,
    TABLES,
    UnknownPluginError,
    load_builtin_plugins,
    register_query,
    register_query_mechanism,
    register_table,
)

__all__ = [
    "QUERIES",
    "QUERY_MECHANISMS",
    "REGISTRIES",
    "TABLES",
    "UnknownPluginError",
    "load_builtin_plugins",
    "register_query",
    "register_query_mechanism",
    "register_table",
]
