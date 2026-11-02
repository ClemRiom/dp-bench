"""Plugin registries.

Every kind of plugin has a registry. For machine learning: mechanisms (training
algorithms), datasets, models, accountants and metrics. For classic DP on databases:
tables, queries and query mechanisms. A plugin registers itself with a decorator::

    @register_mechanism("dp_adam")
    class DPAdam(DPSGD):
        ...

Configuration files, the command line and the dashboard refer to plugins by these names
only, so a new plugin is usable everywhere as soon as its module is imported. Built-in
plugins are imported lazily the first time a registry is queried; external plugins are
loaded with :func:`load_plugins` (a module name or the path of a ``.py`` file). The
plugins registered by each file are remembered, so that a file edited in the dashboard
can be reloaded (``reload=True``) or removed (:func:`unload_plugin_file`).
"""

from __future__ import annotations

import difflib
import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")

REQUIRED = "<required>"

BUILTIN_PACKAGES = (
    "dpbench.datasets",
    "dpbench.models",
    "dpbench.accounting",
    "dpbench.mechanisms",
    "dpbench.metrics",
    "dpbench.classic",
)

_builtin_state = "unloaded"
_loaded_files: dict[Path, str] = {}
#: (kind, name) of the plugins registered by each plugin file.
_registered_by_file: dict[Path, list[tuple[str, str]]] = {}
#: The plugin file being imported, if any (to attribute registrations to it).
_current_file: Path | None = None


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


def plugin_parameters(obj: object, reserved: Iterable[str] = ()) -> dict[str, Any]:
    """Constructor parameters of a plugin and their defaults (``REQUIRED`` when there is none)."""
    defaults = getattr(obj, "default_params", None)
    if callable(defaults):
        return dict(defaults())
    try:
        signature = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    skipped = set(reserved)
    params: dict[str, Any] = {}
    for p in signature.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.name in skipped:
            continue
        default = REQUIRED if p.default is p.empty else p.default
        params[p.name] = list(default) if isinstance(default, tuple) else default
    return params


def accepts_parameter(obj: object, name: str) -> bool:
    """Whether a plugin constructor accepts a keyword argument (or ``**kwargs``)."""
    try:
        signature = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    for p in signature.parameters.values():
        if p.kind == p.VAR_KEYWORD or (p.name == name and p.kind != p.VAR_POSITIONAL):
            return True
    return False


class Registry(Generic[T]):
    """A name -> plugin mapping for one kind of plugin."""

    def __init__(self, kind: str, reserved: Iterable[str] = ()) -> None:
        self.kind = kind
        self.reserved = tuple(reserved)
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
            if _current_file is not None:
                _registered_by_file.setdefault(_current_file, []).append((self.kind, name))
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

    def parameters(self, name: str) -> dict[str, Any]:
        return plugin_parameters(self.get(name), self.reserved)

    def describe(self, name: str) -> dict[str, Any]:
        obj = self.get(name)
        doc = inspect.getdoc(obj) or ""
        return {
            "kind": self.kind,
            "name": name,
            "summary": doc.split("\n\n")[0].replace("\n", " ") if doc else "",
            "doc": doc,
            "parameters": self.parameters(name),
            "module": getattr(obj, "__module__", ""),
        }

    def check_params(self, name: str, params: dict[str, Any]) -> None:
        """Raise ``ValueError`` if ``params`` contains a keyword the plugin does not accept."""
        obj = self.get(name)
        accepted = [p for p in plugin_parameters(obj, self.reserved)]
        for key in params:
            if key in self.reserved or not accepts_parameter(obj, key):
                close = difflib.get_close_matches(key, accepted, n=1)
                hint = f" Did you mean {close[0]!r}?" if close else ""
                listing = ", ".join(accepted) if accepted else "none"
                raise ValueError(
                    f"{self.kind} {name!r} has no parameter {key!r}.{hint} Accepted parameters: {listing}."
                )


MECHANISMS: Registry[Any] = Registry("mechanism")
DATASETS: Registry[Any] = Registry("dataset")
MODELS: Registry[Any] = Registry("model", reserved=("input_shape", "n_classes"))
ACCOUNTANTS: Registry[Any] = Registry("accountant")
METRICS: Registry[Any] = Registry("metric")
TABLES: Registry[Any] = Registry("table")
QUERIES: Registry[Any] = Registry("query")
QUERY_MECHANISMS: Registry[Any] = Registry("query_mechanism")

REGISTRIES: dict[str, Registry[Any]] = {
    "mechanism": MECHANISMS,
    "dataset": DATASETS,
    "model": MODELS,
    "accountant": ACCOUNTANTS,
    "metric": METRICS,
    "table": TABLES,
    "query": QUERIES,
    "query_mechanism": QUERY_MECHANISMS,
}

register_mechanism = MECHANISMS.register
register_dataset = DATASETS.register
register_model = MODELS.register
register_accountant = ACCOUNTANTS.register
register_metric = METRICS.register
register_table = TABLES.register
register_query = QUERIES.register
register_query_mechanism = QUERY_MECHANISMS.register


def plugin_file(spec: str, base_dir: Path | None) -> Path | None:
    """The file a plugin spec points to, or ``None`` if the spec is a module name."""
    if not (spec.endswith(".py") or "/" in spec or "\\" in spec):
        return None
    candidate = Path(spec).expanduser()
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = ([base_dir / candidate] if base_dir is not None else []) + [Path.cwd() / candidate]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    looked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Plugin file {spec!r} not found (looked in: {looked})")


def unload_plugin_file(path: str | Path) -> list[tuple[str, str]]:
    """Unregister the plugins of a plugin file and forget the file. Returns their (kind, name)."""
    path = Path(path).resolve()
    removed = _registered_by_file.pop(path, [])
    for kind, name in removed:
        REGISTRIES[kind].unregister(name)
    module_name = _loaded_files.pop(path, None)
    if module_name is not None:
        sys.modules.pop(module_name, None)
    return removed


def plugins_of_file(path: str | Path) -> list[tuple[str, str]]:
    """(kind, name) of the plugins registered by a plugin file that was loaded."""
    return list(_registered_by_file.get(Path(path).resolve(), []))


def _import_file(path: Path, reload: bool = False) -> str:
    global _current_file
    if path in _loaded_files:
        if not reload:
            return _loaded_files[path]
        unload_plugin_file(path)
    digest = hashlib.sha1(str(path).encode()).hexdigest()[:8]
    module_name = f"dpbench_plugin_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plugin file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    _current_file = path
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        for kind, name in _registered_by_file.pop(path, []):  # undo a partial import
            REGISTRIES[kind].unregister(name)
        raise
    finally:
        _current_file = None
    _loaded_files[path] = module_name
    return module_name


def load_plugins(specs: Iterable[str], base_dir: str | Path | None = None, reload: bool = False) -> list[str]:
    """Import external plugins given as module names or ``.py`` file paths.

    Relative file paths are looked up in ``base_dir`` (typically the directory of the
    configuration file) and then in the current directory. With ``reload=True``, a file
    that was already imported is imported again (its previous plugins are replaced).
    Returns the module names.
    """
    load_builtin_plugins()
    base = Path(base_dir) if base_dir is not None else None
    modules = []
    for spec in specs:
        path = plugin_file(str(spec), base)
        if path is None:
            importlib.import_module(str(spec))
            modules.append(str(spec))
        else:
            modules.append(_import_file(path, reload))
    return modules
