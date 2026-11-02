"""The user's workspace: datasets, algorithms and external mechanisms added from the dashboard.

::

    workspace/
    ├── datasets/   <name>.csv + <name>.json (schema, target column, description)
    ├── plugins/    Python plugin files: training algorithms, query mechanisms, datasets...
    └── external/   mechanisms written in another language: <name>.<ext> + <name>.json

The workspace is the folder ``workspace`` of the current directory, or ``$DPBENCH_WORKSPACE``.
It is loaded when the dashboard starts and before every experiment, so what is added in
the dashboard is usable everywhere: in the forms and in the experiments run in the background.

Plugins are Python code run with the user's rights, exactly like a script: only add code
you trust.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dpbench.core.registry import (
    DATASETS,
    QUERY_MECHANISMS,
    TABLES,
    load_builtin_plugins,
    load_plugins,
    plugins_of_file,
    unload_plugin_file,
)
from dpbench.core.schema import TableSchema

logger = logging.getLogger(__name__)

WORKSPACE_ENV = "DPBENCH_WORKSPACE"
NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Errors of the last load, per file name.
_load_errors: dict[str, str] = {}


def workspace_dir() -> Path:
    """``$DPBENCH_WORKSPACE``, or ``./workspace``."""
    return Path(os.environ.get(WORKSPACE_ENV) or Path.cwd() / "workspace").expanduser().resolve()


def _folder(name: str) -> Path:
    path = workspace_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir() -> Path:
    return _folder("datasets")


def plugins_dir() -> Path:
    return _folder("plugins")


def external_dir() -> Path:
    return _folder("external")


def check_name(name: str) -> str:
    """A plugin name must be a Python identifier (letters, digits and _, not starting with a digit)."""
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"{name!r} is not a valid name: use letters, digits and _, starting with a letter")
    return name


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------


def _is_uploaded(registry, name: str) -> bool:
    from dpbench.classic.tables import UploadedTable
    from dpbench.datasets.uploaded import UploadedDataset

    if name not in registry:
        return True
    return issubclass(registry.get(name), (UploadedTable, UploadedDataset))


def _register_dataset(meta: dict[str, Any], csv_path: Path) -> list[str]:
    """Register an uploaded file as a table (classic DP) and, if it has a target, as a dataset (machine learning)."""
    from dpbench.classic.tables import register_uploaded_table
    from dpbench.datasets.uploaded import register_uploaded_dataset

    name, description = meta["name"], meta.get("description", "")
    register_uploaded_table(name, csv_path, meta["schema"], description)
    kinds = ["table"]
    if meta["schema"].get("target"):
        register_uploaded_dataset(name, csv_path, meta["schema"], description)
        kinds.append("dataset")
    elif _is_uploaded(DATASETS, name) and name in DATASETS:
        DATASETS.unregister(name)
    return kinds


def save_dataset(
    name: str,
    frame: pd.DataFrame,
    schema: TableSchema,
    description: str = "",
    source_file: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save an uploaded table in the workspace and register it. Returns its metadata."""
    load_builtin_plugins()
    check_name(name)
    for registry in (DATASETS, TABLES):
        if not _is_uploaded(registry, name):
            raise ValueError(f"{name!r} is the name of a built-in {registry.kind}; choose another name")
    csv_path, meta_path = datasets_dir() / f"{name}.csv", datasets_dir() / f"{name}.json"
    if meta_path.exists() and not overwrite:
        raise FileExistsError(f"a dataset named {name!r} already exists in the workspace")
    if schema.target is not None and schema.target not in schema.columns:
        raise ValueError(f"unknown target column {schema.target!r}")
    frame.to_csv(csv_path, index=False)
    meta = {
        "name": name,
        "description": description,
        "source_file": source_file,
        "rows": int(len(frame)),
        "created": _now(),
        "schema": schema.to_dict(),
    }
    _write_json(meta_path, meta)
    meta["registered_as"] = _register_dataset(meta, csv_path)
    return meta


def list_datasets() -> list[dict[str, Any]]:
    out = []
    for meta_path in sorted(datasets_dir().glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["path"] = str(meta_path.with_suffix(".csv"))
            out.append(meta)
        except (OSError, ValueError) as error:
            out.append({"name": meta_path.stem, "error": str(error), "path": str(meta_path)})
    return out


def delete_dataset(name: str) -> None:
    for registry in (DATASETS, TABLES):
        if name in registry and _is_uploaded(registry, name):
            registry.unregister(name)
    for suffix in (".csv", ".json"):
        (datasets_dir() / f"{name}{suffix}").unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------
# Python plugins
# ---------------------------------------------------------------------------------------------


def plugin_path(filename: str) -> Path:
    stem = filename[:-3] if filename.endswith(".py") else filename
    check_name(stem)
    return plugins_dir() / f"{stem}.py"


def save_plugin(filename: str, code: str) -> dict[str, Any]:
    """Write a Python plugin file in the workspace and (re)load it. Returns the plugins it registered."""
    path = plugin_path(filename)
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    path.write_text(code, encoding="utf-8")
    try:
        load_plugins([str(path)], reload=True)
    except BaseException:
        if previous is None:  # do not leave a broken new file behind
            path.unlink(missing_ok=True)
        else:
            path.write_text(previous, encoding="utf-8")
            load_plugins([str(path)], reload=True)
        raise
    _load_errors.pop(path.name, None)
    return {"file": path.name, "path": str(path), "plugins": [f"{k}:{n}" for k, n in plugins_of_file(path)]}


def list_plugins() -> list[dict[str, Any]]:
    out = []
    for path in sorted(plugins_dir().glob("*.py")):
        out.append(
            {
                "file": path.name,
                "path": str(path),
                "plugins": [f"{k}:{n}" for k, n in plugins_of_file(path)],
                "error": _load_errors.get(path.name, ""),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return out


def delete_plugin(filename: str) -> None:
    path = plugin_path(filename)
    unload_plugin_file(path)
    path.unlink(missing_ok=True)
    _load_errors.pop(path.name, None)


# ---------------------------------------------------------------------------------------------
# Mechanisms written in another language
# ---------------------------------------------------------------------------------------------


def save_external(
    name: str, language: str, code: str, description: str = "", uses_delta: bool = False
) -> dict[str, Any]:
    """Write an external mechanism (program + manifest) in the workspace and register it."""
    from dpbench.classic.external import LANGUAGES, ExternalMechanism, register_external

    load_builtin_plugins()
    check_name(name)
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}; choose one of {', '.join(LANGUAGES)}")
    if name in QUERY_MECHANISMS and not issubclass(QUERY_MECHANISMS.get(name), ExternalMechanism):
        raise ValueError(f"{name!r} is already the name of a query mechanism; choose another name")
    program = external_dir() / f"{name}{LANGUAGES[language].extension}"
    program.write_text(code, encoding="utf-8")
    manifest = {
        "name": name,
        "language": language,
        "program": program.name,
        "description": description,
        "uses_delta": bool(uses_delta),
        "created": _now(),
    }
    _write_json(external_dir() / f"{name}.json", manifest)
    register_external(manifest, program)
    return manifest


def list_external() -> list[dict[str, Any]]:
    out = []
    for manifest_path in sorted(external_dir().glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["path"] = str(external_dir() / manifest["program"])
            out.append(manifest)
        except (OSError, ValueError, KeyError) as error:
            out.append({"name": manifest_path.stem, "error": str(error)})
    return out


def delete_external(name: str) -> None:
    from dpbench.classic.external import ExternalMechanism

    if name in QUERY_MECHANISMS and issubclass(QUERY_MECHANISMS.get(name), ExternalMechanism):
        QUERY_MECHANISMS.unregister(name)
    manifest_path = external_dir() / f"{name}.json"
    if manifest_path.exists():
        try:
            program = json.loads(manifest_path.read_text(encoding="utf-8")).get("program")
            if program:
                (external_dir() / program).unlink(missing_ok=True)
        except ValueError:
            pass
        manifest_path.unlink()


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def load_workspace(reload: bool = False) -> dict[str, str]:
    """Register everything in the workspace. Returns the errors (file name -> message); never raises."""
    from dpbench.classic.external import register_external

    load_builtin_plugins()
    root = workspace_dir()
    if not root.is_dir():
        return {}
    errors: dict[str, str] = {}
    for meta in list_datasets():
        try:
            if "error" in meta:
                raise ValueError(meta["error"])
            _register_dataset(meta, Path(meta["path"]))
        except Exception as error:
            errors[f"datasets/{meta['name']}"] = f"{type(error).__name__}: {error}"
    for path in sorted(plugins_dir().glob("*.py")):
        try:
            load_plugins([str(path)], reload=reload)
        except Exception as error:
            errors[path.name] = f"{type(error).__name__}: {error}"
    for manifest in list_external():
        try:
            if "error" in manifest:
                raise ValueError(manifest["error"])
            register_external(manifest, Path(manifest["path"]))
        except Exception as error:
            errors[f"external/{manifest['name']}"] = f"{type(error).__name__}: {error}"
    _load_errors.clear()
    _load_errors.update({k: v for k, v in errors.items() if k.endswith(".py")})
    for name, message in errors.items():
        logger.warning("workspace: %s could not be loaded: %s", name, message)
    return errors
