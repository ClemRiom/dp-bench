"""Check a plugin file before adding it to the workspace: import it, then try every plugin it registers.

The dashboard runs this module in a separate process (``python -m dpbench.plugin_check
FILE``), so that a broken or slow plugin cannot crash or freeze the dashboard; the report
is printed as JSON. Each kind of plugin gets a small smoke test:

* training algorithm: two epochs on 300 synthetic records, then predictions;
* query mechanism: one release of a small task of every kind it supports, then 300
  releases of a count to measure its noise;
* dataset or table: loading it;
* model, metric, accountant: building and calling them once.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _short_traceback(limit: int = 12) -> str:
    return "".join(traceback.format_exc().splitlines(keepends=True)[-limit:])


def _test_mechanism(cls) -> dict[str, Any]:
    from dpbench.core.registry import MODELS
    from dpbench.datasets.synthetic import Synthetic

    data = Synthetic(n_samples=300, n_features=6, seed=0).load()
    model = MODELS.get("mlp")(data.input_shape, data.n_classes)
    mechanism = cls()
    fit = mechanism.fit(
        model, data.X_train, data.y_train, target_epsilon=2.0, delta=1e-5, epochs=2, batch_size=32, seed=0
    )
    proba = mechanism.predict_proba(data.X_test)
    if proba.shape != (data.n_test, data.n_classes) or not np.allclose(proba.sum(axis=1), 1, atol=1e-4):
        raise ValueError(f"predict_proba returned an array of shape {proba.shape} that is not a probability table")
    epsilon = float(fit.epsilon)
    if cls.private and not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError(f"a private mechanism must report a finite epsilon spent, got {epsilon}")
    return {
        "test": "2 epochs on 300 synthetic records (target ε = 2)",
        "accuracy": float(np.mean(proba.argmax(axis=1) == data.y_test)),
        "epsilon_spent": epsilon,
        "noise_multiplier": fit.noise_multiplier,
    }


def _test_query_mechanism(cls) -> dict[str, Any]:
    from dpbench.classic.tasks import NumericTask, QuantileTask, SelectionTask, WorkloadTask

    rng = np.random.default_rng(0)
    examples = {
        "numeric": NumericTask([10.0, 20.0, 30.0], 1.0, 1.0, integer=True),
        "selection": SelectionTask([5.0, 50.0, 3.0], 1.0, ["a", "b", "c"]),
        "quantile": QuantileTask(np.arange(100.0), 0.0, 100.0, 0.5),
        "workload": WorkloadTask(np.full(16, 10.0), np.triu(np.ones((16, 16)))[:5]),
    }
    mechanism = cls()
    outputs = {}
    for kind in cls.supports:
        task = examples[kind]
        mechanism.check(task, 1.0, 1e-5)
        output = mechanism.release(task, 1.0, 1e-5, rng)
        if kind in ("numeric", "workload") and np.shape(output) != np.shape(task.answer):
            raise ValueError(
                f"{kind}: released an answer of shape {np.shape(output)} instead of {np.shape(task.answer)}"
            )
        outputs[kind] = np.asarray(output, dtype=float).round(3).tolist()
    report: dict[str, Any] = {"test": "one release of each supported task kind (ε = 1)", "outputs": outputs}
    if "numeric" in cls.supports:
        count = NumericTask([1000.0], 1.0, 1.0, integer=True)
        noise = np.array([mechanism.release(count, 1.0, 1e-5, rng)[0] - 1000.0 for _ in range(300)])
        report.update(noise_mean=float(noise.mean()), noise_std=float(noise.std()), laplace_std=math.sqrt(2.0))
    return report


def _test_loader(cls) -> dict[str, Any]:
    loaded = cls().load()
    size = getattr(loaded, "n_train", None) or getattr(loaded, "n", None)
    return {"test": "load()", "records": int(size or 0), "type": type(loaded).__name__}


def _test_model(cls) -> dict[str, Any]:
    import torch

    model = cls((6,), 2)
    output = model(torch.zeros(4, 6))
    return {"test": "forward pass on 4 x 6 zeros", "output_shape": list(output.shape)}


def _test_metric(cls) -> dict[str, Any]:
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]])
    return {"test": "on 4 predictions", "value": float(cls()(y, proba, None))}


def _test_accountant(cls) -> dict[str, Any]:
    return {"test": "σ = 1, q = 0.01, 1000 steps", "epsilon": float(cls().epsilon(1.0, 0.01, 1000, 1e-5))}


TESTS = {
    "mechanism": _test_mechanism,
    "query_mechanism": _test_query_mechanism,
    "dataset": _test_loader,
    "table": _test_loader,
    "model": _test_model,
    "metric": _test_metric,
    "accountant": _test_accountant,
}


def check_plugin_file(path: str | Path) -> dict[str, Any]:
    """Import a plugin file and smoke-test every plugin it registers (in the current process)."""
    from dpbench.core.registry import REGISTRIES, load_builtin_plugins, load_plugins, plugins_of_file

    path = Path(path).resolve()
    report: dict[str, Any] = {"file": str(path), "ok": False, "plugins": []}
    load_builtin_plugins()
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as error:
        report["error"] = f"SyntaxError line {error.lineno}: {error.msg}"
        report["line"] = error.lineno
        return report
    try:
        load_plugins([str(path)])
    except Exception as error:
        report["error"] = f"{type(error).__name__} while importing the file: {error}"
        report["traceback"] = _short_traceback()
        return report
    registered = plugins_of_file(path)
    if not registered:
        report["error"] = (
            "the file defines no plugin: decorate your class with a register decorator, "
            'e.g. @register_mechanism("my_algorithm") or @register_query_mechanism("my_mechanism")'
        )
        return report
    all_ok = True
    for kind, name in registered:
        cls = REGISTRIES[kind].get(name)
        entry: dict[str, Any] = {"kind": kind, "name": name, "summary": (cls.__doc__ or "").strip().split("\n")[0]}
        start = time.perf_counter()
        try:
            test = TESTS.get(kind)
            entry.update(test(cls) if test else {"test": "registered (no automatic test for this kind)"})
            entry["ok"] = True
        except Exception as error:
            entry.update(ok=False, error=f"{type(error).__name__}: {error}", traceback=_short_traceback())
            all_ok = False
        entry["seconds"] = round(time.perf_counter() - start, 3)
        report["plugins"].append(entry)
    report["ok"] = all_ok
    return report


def check_plugin_code(code: str, filename: str, timeout: float = 300.0) -> dict[str, Any]:
    """Write ``code`` to a temporary file and check it in a separate Python process."""
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="dpbench-check-"))
    path = directory / filename
    path.write_text(code, encoding="utf-8")
    try:
        process = subprocess.run(
            [sys.executable, "-m", "dpbench.plugin_check", str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the check did not finish within {timeout:g} s (infinite loop?)", "plugins": []}
    finally:
        path.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass
    lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {
            "ok": False,
            "error": "the check crashed:\n" + (process.stderr or process.stdout)[-2000:],
            "plugins": [],
        }
    report = json.loads(lines[-1])
    report["file"] = filename
    return report


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m dpbench.plugin_check PLUGIN_FILE.py", file=sys.stderr)
        return 2
    import logging
    import warnings

    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    report = check_plugin_file(args[0])
    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
