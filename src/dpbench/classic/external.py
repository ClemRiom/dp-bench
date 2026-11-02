"""Query mechanisms written in another language: DP-Bench runs them as external programs.

The protocol is deliberately minimal, so that a mechanism can be written in any language
without a JSON library::

    <interpreter> <program file> EPSILON DELTA L1_SENSITIVITY L2_SENSITIVITY SEED

* the exact answer arrives on the **standard input**, one number per line;
* the program writes the noisy answer on the **standard output**, one number per line
  (as many as it received).

External mechanisms add noise to numeric answers (counts, sums, histograms, workloads
answered query by query). Training algorithms stay in Python: they need the model, the
gradients and PyTorch, which only exist inside the Python process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from dpbench.classic.mechanisms import QueryMechanism, numeric_view
from dpbench.classic.tasks import NUMERIC, WORKLOAD
from dpbench.core.registry import QUERY_MECHANISMS, register_query_mechanism


class ExternalMechanismError(RuntimeError):
    """The external program failed or did not follow the protocol."""


@dataclass(frozen=True)
class Language:
    title: str
    extension: str
    #: Command line; ``{file}`` is replaced by the program file.
    command: tuple[str, ...]
    template: str

    def resolved(self, file: Path) -> list[str]:
        return [str(file) if part == "{file}" else part for part in self.command]

    @property
    def available(self) -> bool:
        program = self.command[0]
        return Path(program).is_file() or shutil.which(program) is not None


PYTHON_TEMPLATE = '''"""Laplace mechanism as an external program (DP-Bench protocol).

Arguments: EPSILON DELTA L1_SENSITIVITY L2_SENSITIVITY SEED.
Standard input: the exact answer, one number per line. Standard output: the noisy answer.
"""
import random
import sys

epsilon, delta, l1, l2 = (float(v) for v in sys.argv[1:5])
random.seed(int(sys.argv[5]))
scale = l1 / epsilon  # Laplace scale b = sensitivity / epsilon

for line in sys.stdin:
    if line.strip():
        # A Laplace(0, b) variable is the difference of two exponential variables of mean b.
        noise = random.expovariate(1 / scale) - random.expovariate(1 / scale)
        print(repr(float(line) + noise))
'''

JAVASCRIPT_TEMPLATE = """// Laplace mechanism as an external program (DP-Bench protocol).
// Arguments: EPSILON DELTA L1_SENSITIVITY L2_SENSITIVITY SEED.
// Standard input: the exact answer, one number per line. Standard output: the noisy answer.
const [epsilon, delta, l1, l2, seed] = process.argv.slice(2).map(Number);

// Seeded generator (mulberry32), so that a run can be reproduced.
let state = seed >>> 0;
function random() {
  state = (state + 0x6d2b79f5) >>> 0;
  let t = Math.imul(state ^ (state >>> 15), state | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

const scale = l1 / epsilon; // Laplace scale b = sensitivity / epsilon
const exponential = () => -scale * Math.log(1 - random());
const lines = require("fs").readFileSync(0, "utf8").split("\\n").filter((line) => line.trim() !== "");
// A Laplace(0, b) variable is the difference of two exponential variables of mean b.
console.log(lines.map((line) => Number(line) + exponential() - exponential()).join("\\n"));
"""

R_TEMPLATE = """# Laplace mechanism as an external program (DP-Bench protocol).
# Arguments: EPSILON DELTA L1_SENSITIVITY L2_SENSITIVITY SEED.
# Standard input: the exact answer, one number per line. Standard output: the noisy answer.
args <- as.numeric(commandArgs(trailingOnly = TRUE))
epsilon <- args[1]
l1 <- args[3]
set.seed(args[5])
answer <- as.numeric(readLines(file("stdin")))
scale <- l1 / epsilon  # Laplace scale b = sensitivity / epsilon
# A Laplace(0, b) variable is the difference of two exponential variables of mean b.
noisy <- answer + rexp(length(answer), 1 / scale) - rexp(length(answer), 1 / scale)
cat(sprintf("%.17g", noisy), sep = "\\n")
"""

JAVA_TEMPLATE = """// Laplace mechanism as an external program (DP-Bench protocol), run with "java <file>" (Java 11+).
// Arguments: EPSILON DELTA L1_SENSITIVITY L2_SENSITIVITY SEED.
// Standard input: the exact answer, one number per line. Standard output: the noisy answer.
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;

public class Mechanism {
    public static void main(String[] args) throws Exception {
        double epsilon = Double.parseDouble(args[0]);
        double l1 = Double.parseDouble(args[2]);
        Random random = new Random(Long.parseLong(args[4]));
        double scale = l1 / epsilon; // Laplace scale b = sensitivity / epsilon
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            // A Laplace(0, b) variable is the difference of two exponential variables of mean b.
            double noise = -scale * Math.log(1 - random.nextDouble()) + scale * Math.log(1 - random.nextDouble());
            System.out.println(Double.parseDouble(line) + noise);
        }
    }
}
"""

BASH_TEMPLATE = """#!/usr/bin/env bash
# Laplace mechanism as an external program (DP-Bench protocol), with awk for the arithmetic.
# Arguments: EPSILON DELTA L1_SENSITIVITY L2_SENSITIVITY SEED.
# Standard input: the exact answer, one number per line. Standard output: the noisy answer.
awk -v epsilon="$1" -v l1="$3" -v seed="$5" '
function uniform() {   # Park-Miller generator: the same numbers for a given seed with every awk
    state = (16807 * state) % 2147483647
    return state / 2147483647              # strictly between 0 and 1
}
BEGIN {
    state = seed % 2147483646 + 1
    for (i = 0; i < 10; i++) uniform()     # the first numbers of a small seed are not random enough
    scale = l1 / epsilon                   # Laplace scale b = sensitivity / epsilon
}
NF {
    u = uniform() - 0.5                    # inverse CDF of the Laplace distribution
    sign = (u < 0) ? -1 : 1
    printf "%.17g\\n", $1 - scale * sign * log(1 - 2 * sign * u)
}'
"""

LANGUAGES: dict[str, Language] = {
    "python": Language("Python", ".py", (sys.executable, "{file}"), PYTHON_TEMPLATE),
    "javascript": Language("JavaScript (Node.js)", ".js", ("node", "{file}"), JAVASCRIPT_TEMPLATE),
    "r": Language("R", ".R", ("Rscript", "{file}"), R_TEMPLATE),
    "java": Language("Java (single file, Java 11+)", ".java", ("java", "{file}"), JAVA_TEMPLATE),
    "bash": Language("Bash + awk", ".sh", ("bash", "{file}"), BASH_TEMPLATE),
}


def run_program(
    command: list[str],
    answer: np.ndarray,
    epsilon: float,
    delta: float,
    l1: float,
    l2: float,
    seed: int,
    timeout: float = 60.0,
) -> np.ndarray:
    """Run an external mechanism once with the DP-Bench protocol and parse its output."""
    arguments = [repr(float(epsilon)), repr(float(delta)), repr(float(l1)), repr(float(l2)), str(int(seed))]
    stdin = "\n".join(repr(float(v)) for v in np.ravel(answer)) + "\n"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        process = subprocess.run(
            command + arguments, input=stdin, capture_output=True, text=True, timeout=timeout, env=env
        )
    except FileNotFoundError as error:
        raise ExternalMechanismError(f"interpreter not found: {command[0]!r} ({error})") from error
    except subprocess.TimeoutExpired as error:
        raise ExternalMechanismError(f"the program did not finish within {timeout:g} s") from error
    if process.returncode != 0:
        raise ExternalMechanismError(
            f"the program exited with code {process.returncode}: {process.stderr.strip()[-800:] or '(no error output)'}"
        )
    try:
        values = np.array([float(line) for line in process.stdout.split() if line.strip()], dtype=np.float64)
    except ValueError as error:
        raise ExternalMechanismError(f"the output is not one number per line: {process.stdout[:300]!r}") from error
    if values.size != np.size(answer):
        raise ExternalMechanismError(f"the program wrote {values.size} numbers instead of {np.size(answer)}")
    return values.reshape(np.shape(answer))


class ExternalMechanism(QueryMechanism):
    """Base class of the mechanisms run as external programs (one class per program, see :func:`register_external`)."""

    supports = (NUMERIC, WORKLOAD)
    command: ClassVar[list[str]] = []
    language: ClassVar[str] = ""
    program: ClassVar[str] = ""
    timeout: ClassVar[float] = 60.0

    def release(self, task, epsilon, delta, rng):
        answer, l1, l2, _ = numeric_view(task)
        seed = int(rng.integers(0, 2**31 - 1))
        return run_program(list(self.command), answer, epsilon, delta, l1, l2, seed, self.timeout)


def register_external(manifest: dict[str, Any], program: Path) -> type:
    """Register (or replace) the query mechanism of an external program described by a manifest.

    The manifest has a ``name``, a ``language`` (a key of :data:`LANGUAGES`), a
    ``description`` and ``uses_delta`` (true for an (ε, δ)-DP mechanism).
    """
    language = LANGUAGES[manifest["language"]]
    name = manifest["name"]
    description = manifest.get("description") or f"External mechanism written in {language.title}."
    cls = type(
        f"External_{name}",
        (ExternalMechanism,),
        {
            "__doc__": description,
            "title": manifest.get("title") or name,
            "command": language.resolved(Path(program)),
            "language": manifest["language"],
            "program": str(program),
            "uses_delta": bool(manifest.get("uses_delta", False)),
        },
    )
    QUERY_MECHANISMS.unregister(name)
    return register_query_mechanism(name)(cls)


def check_program(language: str, program: Path, n_values: int = 2000, epsilon: float = 1.0) -> dict[str, Any]:
    """Run a program on test inputs and describe what it does (used before saving it).

    The answer is ``n_values`` copies of 100 with sensitivity 1: the output should be
    centred on 100 and, for the Laplace template, have a standard deviation of √2/ε.
    """
    command = LANGUAGES[language].resolved(program)
    report: dict[str, Any] = {"ok": False, "language": language, "command": " ".join(command)}
    answer = np.full(n_values, 100.0)
    try:
        start = time.perf_counter()
        first = run_program(command, answer, epsilon, 1e-6, 1.0, 1.0, seed=1)
        report["seconds_per_call"] = time.perf_counter() - start
        second = run_program(command, answer, epsilon, 1e-6, 1.0, 1.0, seed=2)
        again = run_program(command, answer, epsilon, 1e-6, 1.0, 1.0, seed=1)
    except ExternalMechanismError as error:
        report["error"] = str(error)
        return report
    noise = first - answer
    report.update(
        ok=bool(np.all(np.isfinite(first))),
        noise_mean=float(noise.mean()),
        noise_std=float(noise.std()),
        laplace_std=float(np.sqrt(2) / epsilon),
        random=bool(np.std(noise) > 0 and not np.array_equal(first, second)),
        reproducible=bool(np.array_equal(first, again)),
        sample=[float(v) for v in first[:5]],
    )
    if not report["ok"]:
        report["error"] = "the program returned non-finite numbers"
    return report
