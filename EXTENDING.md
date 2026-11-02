# Adding your own datasets, mechanisms and accountants

Everything is a **plugin**: a class registered under a name with a decorator. The dashboard,
the benchmark and the configuration only use these names, so a new plugin appears in every
list as soon as its file is imported.

Two ways to add one, with the same code:

* **From the dashboard** (no change to DP-Bench): *Add a dataset* for a data file, *Add an
  algorithm* for code. *Check the code* imports the file in a separate process and runs a short
  test; *Save* copies it into `workspace/` (loaded at start-up and before every experiment).
* **In the package**: put the class in the right module (for instance
  `src/dpbench/classic/mechanisms.py`); it is registered when the package is imported.

Every template of the *Add an algorithm* page is a complete, working plugin.

## A dataset

**Without code**: drop a CSV, TSV, Excel, Parquet or JSON file on *Add a dataset*, check the
type and the public bounds of every column, pick the target column (to train on it) and,
optionally, a sensitive attribute (for the fairness metrics). The file becomes a *table* for
the queries and, with a target, a *dataset* for training.

**With code**: a subclass of `DatasetLoader` whose `load_full()` returns a `Dataset`.

```python
from dpbench.core.registry import register_dataset
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader


@register_dataset("my_data")
class MyData(DatasetLoader):
    """One line shown in the lists."""

    def load_full(self) -> Dataset:
        ...  # read, encode, scale to [0, 1], split
        return Dataset(
            name="my_data",
            X_train=X_train, y_train=y_train,  # float32 (n, d) or (n, c, h, w); int64 labels 0..k-1
            X_test=X_test, y_test=y_test,
            n_classes=k,
            groups_train=None, groups_test=None,  # optional: integer code of a sensitive attribute
        )
```

## A query mechanism (classic DP)

### As one function (additive noise)

```python
from dpbench.classic import noise_mechanism


@noise_mechanism("my_laplace", title="My Laplace")
def my_laplace(answer, sensitivity, epsilon, delta, rng):
    return answer + rng.laplace(0.0, sensitivity / epsilon, size=answer.shape)
```

| Argument | What it is |
| --- | --- |
| `answer` | the exact answer, a float64 vector (a count is a vector of length 1) |
| `sensitivity` | how much one record (added or removed) can change it: L1 norm, or L2 with `sensitivity="l2"` |
| `epsilon`, `delta` | the budget of this release (`delta` only matters with `uses_delta=True`) |
| `rng` | a `numpy.random.Generator`: draw all the noise from it, so runs are reproducible |

Return the noisy answer, same shape. Options of the decorator: `title`, `sensitivity="l1" | "l2"`,
`uses_delta=False | True`. The mechanism answers counts, sums, means, histograms, marginals,
centroids and range queries answered one by one.

### As a class (any kind of mechanism)

```python
from dpbench.classic import QueryMechanism, numeric_view
from dpbench.core.registry import register_query_mechanism


@register_query_mechanism("my_mechanism")
class MyMechanism(QueryMechanism):
    """One line shown in the lists."""

    title = "My mechanism"
    supports = ("numeric", "workload")  # task kinds: numeric, selection, quantile, workload
    uses_delta = False                  # True for an (epsilon, delta)-DP mechanism

    def __init__(self, my_parameter: float = 1.0):  # optional parameters, with defaults
        self.my_parameter = my_parameter

    def release(self, task, epsilon, delta, rng):
        answer, l1, l2, integer = numeric_view(task)  # numeric and workload tasks
        return answer + rng.laplace(0.0, l1 / epsilon, answer.shape)

    def check(self, task, epsilon, delta):  # optional: refuse a configuration (it is then skipped)
        super().check(task, epsilon, delta)
```

What `release` receives, by task kind:

| Kind | Fields | Return |
| --- | --- | --- |
| `numeric` | `answer`, `l1_sensitivity`, `l2_sensitivity`, `integer`, `labels` | a vector of the shape of `answer` |
| `workload` | `histogram`, `workload` (one row per query), `answer = workload @ histogram` | a vector, one value per query |
| `selection` | `scores`, `sensitivity`, `candidates`, `monotone` | the index of the chosen candidate |
| `quantile` | `values` (sorted, clipped), `lower`, `upper`, `q` | a number in `[lower, upper]` |

A query made of several tasks (the mean: a sum and a count) splits its budget equally between
them; the mechanism only sees one task and its share of epsilon.

### In another language

*Add an algorithm → Query mechanism in another language* (JavaScript, Java, R, Bash, Python).
DP-Bench runs `PROGRAM EPSILON DELTA L1_SENSITIVITY L2_SENSITIVITY SEED`, writes the exact
answer on the standard input (one number per line) and reads the noisy answer on the
standard output.

## A training algorithm (machine learning)

Most variants of DP-SGD only change one step. Subclass `DPSGD` and give it a new engine:

```python
from dpbench.core.registry import register_mechanism
from dpbench.mechanisms.dpsgd import DPSGD
from dpbench.mechanisms.engine import PrivateGradient


class MyGradient(PrivateGradient):
    def clip_factors(self, norms):       # factor applied to each per-example gradient
        return (self.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

    # def sample_noise(self, like): ...   the noise added to the clipped sum
    # def privatize(self, x, y, expected_batch_size): ...   the whole private gradient


@register_mechanism("my_dpsgd")
class MyDPSGD(DPSGD):
    """One line shown in the lists."""

    title = "My DP-SGD"
    engine_class = MyGradient
    default_lr = 0.5
    # noise = "laplace"; clipping_norm = 1; default_accountant = "pld"   (see dp_lap_sgd.py)
```

The noise is calibrated by the chosen accountant before training, so the target epsilon holds
as long as the sensitivity of the clipped sum stays `max_grad_norm`. A completely different
algorithm subclasses `Mechanism` and implements `fit(model, X, y, *, target_epsilon, delta,
epochs, batch_size, seed, device, accountant, callback) -> FitResult` (see `non_private.py`).

## A privacy accountant

```python
from dpbench.accounting.base import Accountant
from dpbench.core.registry import register_accountant


@register_accountant("my_accountant")
class MyAccountant(Accountant):
    """One line shown in the lists."""

    supported_noises = ("gaussian",)  # and/or "laplace"
    approximate = False               # True if the result is not a guaranteed upper bound

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        ...  # epsilon of `steps` Poisson-subsampled noisy steps at this delta
```

It then appears on the *Run an experiment* page (Accountants) and on the *Accountants* page,
and the noise of every training that uses it is calibrated with it.

## Other plugins

Tables for the queries (`TableLoader`, `classic/tables.py`), queries (`Query`,
`classic/queries.py`), models (`register_model`, `models/__init__.py`) and metrics (`Metric`,
`metrics/__init__.py`) follow the same pattern: a class, a decorator, a docstring.
