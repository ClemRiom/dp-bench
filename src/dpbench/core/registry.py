"""Plugin registries.

Every kind of plugin has a registry: for classic DP on databases, tables, queries and query
mechanisms. A plugin registers itself with a decorator::

    @register_query("mean")
    class Mean(Query):
        ...

The dashboard and the benchmark refer to plugins by these names only, so a new plugin is
usable everywhere as soon as its module is imported. Built-in plugins are imported lazily
the first time a registry is queried.
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")

BUILTIN_PACKAGES = ("dpbench.classic",)

_builtin_state = "unloaded"


class UnknownPluginError(LookupError):
    """A configuration refers to a plugin name that is not registered."""

    def __init__(self, kind: str, name: str, available: list[str]) -> None:
        self.kind, self.name, self.available = kind, name, available
        close = difflib.get_close_matches(str(name), available, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        listing = ", ".join(available) if available else "none"
        super().__init__(f"Unknown {kind} {name!r}.{hint} Available: {listing}.")

    def __str__(self) -> str:
        return str(self.args[0])


def load_builtin_plugins() -> None:
    """Import the built-in plugin packages once (they register themselves on import)."""
    global _builtin_state
    if _builtin_state != "unloaded":
        return
    _builtin_state = "loading"
    try:
        for package in BUILTIN_PACKAGES:
            importlib.import_module(package)
    except BaseException:
        _builtin_state = "unloaded"
        raise
    _builtin_state = "loaded"


def _qualified_name(obj: object) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"


class Registry(Generic[T]):
    """A name -> plugin mapping for one kind of plugin."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator registering a plugin under ``name``."""
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"{self.kind} names must be valid Python identifiers, got {name!r}")

        def decorator(obj: T) -> T:
            existing = self._entries.get(name)
            if existing is not None and existing is not obj and _qualified_name(existing) != _qualified_name(obj):
                raise ValueError(
                    f"{self.kind} {name!r} is already registered by {_qualified_name(existing)}; choose another name"
                )
            obj.name = name  # type: ignore[attr-defined]
            self._entries[name] = obj
            return obj

        return decorator

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def get(self, name: str) -> T:
        load_builtin_plugins()
        if name not in self._entries:
            raise UnknownPluginError(self.kind, name, sorted(self._entries))
        return self._entries[name]

    def names(self) -> list[str]:
        load_builtin_plugins()
        return sorted(self._entries)

    def items(self) -> list[tuple[str, T]]:
        return [(name, self._entries[name]) for name in self.names()]

    def __contains__(self, name: object) -> bool:
        load_builtin_plugins()
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())


TABLES: Registry[Any] = Registry("table")
QUERIES: Registry[Any] = Registry("query")
QUERY_MECHANISMS: Registry[Any] = Registry("query_mechanism")

REGISTRIES: dict[str, Registry[Any]] = {"table": TABLES, "query": QUERIES, "query_mechanism": QUERY_MECHANISMS}

register_table = TABLES.register
register_query = QUERIES.register
register_query_mechanism = QUERY_MECHANISMS.register
