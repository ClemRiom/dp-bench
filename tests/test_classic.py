"""Classic DP: schemas, queries (exact answers and sensitivities), mechanisms (noise distributions), runner."""

import math

import numpy as np
import pandas as pd
import pytest

from dpbench.accounting.accountants import gdp_delta
from dpbench.classic.mechanisms import analytic_gaussian_sigma
from dpbench.classic.runner import AUTO, ClassicConfig, Spec, generic_queries, run_query_benchmark
from dpbench.classic.tables import Table
from dpbench.classic.tasks import NumericTask, QuantileTask, SelectionTask, WorkloadTask
from dpbench.core.registry import QUERIES, QUERY_MECHANISMS, TABLES
from dpbench.core.schema import CATEGORICAL, NUMERIC, ColumnSpec, TableSchema


@pytest.fixture(scope="module")
def table():
    """10 records: ages 20..65 (step 5), incomes, two cities, one missing income."""
    frame = pd.DataFrame(
        {
            "age": np.arange(20, 70, 5, dtype=float),
            "income": [10, 20, 30, 40, 50, 60, 70, 80, 90, np.nan],
            "city": ["Paris"] * 6 + ["Lyon"] * 4,
        }
    )
    columns = {
        "age": ColumnSpec("age", NUMERIC, lower=0, upper=100),
        "income": ColumnSpec("income", NUMERIC, lower=0, upper=100),
        "city": ColumnSpec("city", CATEGORICAL, categories=["Lyon", "Paris"]),
    }
    return Table("toy", frame, TableSchema(columns))


def release(name, task, epsilon=1.0, delta=1e-6, n=1, seed=0, **params):
    mechanism = QUERY_MECHANISMS.get(name)(**params)
    rng = np.random.default_rng(seed)
    return [mechanism.release(task, epsilon, delta, rng) for _ in range(n)]


# --------------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------------


def test_column_specs_are_checked():
    spec = ColumnSpec("age", NUMERIC, lower=0, upper=100)
    assert spec.magnitude == 100 and spec.width == 100
    assert ColumnSpec("t", NUMERIC, lower=-30, upper=10).magnitude == 30  # sum sensitivity: largest |bound|
    with pytest.raises(ValueError, match="needs a lower and an upper bound"):
        ColumnSpec("age", NUMERIC)
    with pytest.raises(ValueError, match="invalid bounds"):
        ColumnSpec("age", NUMERIC, lower=10, upper=0)
    with pytest.raises(ValueError, match="needs its list of categories"):
        ColumnSpec("city", CATEGORICAL)
    assert ColumnSpec("level", CATEGORICAL, categories=[1, 2]).categories == ["1", "2"]  # stored as text


# --------------------------------------------------------------------------------------------
# Queries: exact answers and sensitivities
# --------------------------------------------------------------------------------------------


def test_count_and_conditions(table):
    (task,) = QUERIES.get("count")().tasks(table)
    assert isinstance(task, NumericTask) and task.answer[0] == 10 and task.l1_sensitivity == 1 and task.integer
    assert QUERIES.get("count")(column="age", operator=">=", value=50).exact(table) == 4
    assert QUERIES.get("count")(column="city", operator="==", value="Lyon").exact(table) == 4
    with pytest.raises(ValueError):
        QUERIES.get("count")(column="age", operator="=>")


def test_sum_mean_sensitivity_is_the_largest_bound(table):
    (task,) = QUERIES.get("sum")(column="income").tasks(table)
    assert task.answer[0] == 450 and task.l1_sensitivity == 100  # max(|0|, |100|)
    mean = QUERIES.get("mean")(column="income")
    tasks = mean.tasks(table)
    assert [t.answer[0] for t in tasks] == [450, 9]  # the missing income is not counted
    assert mean.exact(table, tasks) == pytest.approx(50)
    assert mean.combine([np.array([450.0]), np.array([-3.0])]) <= 100  # post-processing keeps it in the domain


def test_histogram_marginal_centroid(table):
    (hist,) = QUERIES.get("histogram")(column="city").tasks(table)
    assert hist.answer.tolist() == [4, 6] and hist.l1_sensitivity == 1  # categories sorted: Lyon, Paris
    (marginal,) = QUERIES.get("marginal")(columns=["city", "age"], bins=2).tasks(table)
    assert marginal.answer.sum() == 10 and marginal.answer.size == 4
    centroid = QUERIES.get("centroid")(columns=["age", "income"])
    vector, count = centroid.tasks(table)
    assert count.answer[0] == 9  # complete rows only
    assert vector.l1_sensitivity == 200 and vector.l2_sensitivity == pytest.approx(math.sqrt(2) * 100)
    exact = centroid.exact(table)
    assert exact == pytest.approx([40.0, 50.0])
    assert centroid.errors(exact + np.array([3.0, 4.0]), exact, table, [vector, count])["abs_error"] == 5


def test_range_queries_workload(table):
    query = QUERIES.get("range_queries")(column="age", bins=10, kind="prefix")
    (task,) = query.tasks(table)
    assert isinstance(task, WorkloadTask)
    assert task.workload.shape == (10, 10)
    assert task.answer[-1] == 10  # the last prefix covers everything
    assert task.l1_sensitivity == 10  # a record in the first cell is in all 10 prefixes
    random = QUERIES.get("range_queries")(column="age", bins=8, n_queries=20, seed=3)
    assert random.tasks(table)[0].workload.shape == (20, 8)
    assert np.array_equal(random.tasks(table)[0].workload, random.tasks(table)[0].workload)  # public, seeded


def test_quantile_rank_error_ignores_gaps_between_values():
    frame = pd.DataFrame({"x": [1.0] * 40 + [2.0] * 20 + [3.0] * 40})
    table = Table("ties", frame, TableSchema({"x": ColumnSpec("x", NUMERIC, lower=0.0, upper=4.0)}))
    query = QUERIES.get("quantile")(column="x", q=0.5, lower=0, upper=4)
    tasks = query.tasks(table)
    exact = query.exact(table, tasks)
    assert exact == 2.0
    assert query.errors(2.0, exact, table, tasks)["rel_error"] == 0
    assert query.errors(2.5, exact, table, tasks)["rel_error"] == 0  # in the gap next to the median
    assert query.errors(3.0, exact, table, tasks)["rel_error"] == pytest.approx(0.4)  # 40% of the records
    assert query.errors(0.5, exact, table, tasks)["rel_error"] == pytest.approx(0.4)


def test_mode_regret(table):
    query = QUERIES.get("mode")(column="city")
    (task,) = query.tasks(table)
    assert isinstance(task, SelectionTask) and task.monotone
    assert query.exact(table, [task]) == 1  # Paris
    assert query.errors(0, 1, table, [task]) == {"abs_error": 2.0, "rel_error": pytest.approx(2 / 6), "correct": 0.0}


# --------------------------------------------------------------------------------------------
# Mechanisms: the noise has the distribution of the theory
# --------------------------------------------------------------------------------------------


COUNT = NumericTask([1000.0], 1.0, 1.0, integer=True)


def test_only_four_query_mechanisms():
    assert set(QUERY_MECHANISMS.names()) >= {"laplace", "gaussian", "exponential", "polar_laplace"}


def test_laplace_scale():
    noise = np.ravel(release("laplace", COUNT, epsilon=0.5, n=20000)) - 1000
    assert abs(noise.mean()) < 0.1
    assert noise.std() == pytest.approx(math.sqrt(2) * 2.0, rel=0.04)  # Lap(Δ/ε), b = 2


def test_gaussian_analytic_calibration():
    for epsilon in (0.1, 0.5, 1.0, 4.0):
        sigma = analytic_gaussian_sigma(epsilon, 1e-6, 1.0)
        assert gdp_delta(epsilon, 1.0 / sigma) == pytest.approx(1e-6, rel=1e-6)  # exact (ε, δ) profile
        if epsilon < 1:  # always less noise than the textbook σ = Δ √(2 ln(1.25/δ)) / ε
            assert sigma < math.sqrt(2 * math.log(1.25 / 1e-6)) / epsilon
    noise = np.ravel(release("gaussian", COUNT, epsilon=1.0, n=20000)) - 1000
    assert noise.std() == pytest.approx(analytic_gaussian_sigma(1.0, 1e-6, 1.0), rel=0.03)
    with pytest.raises(ValueError):
        QUERY_MECHANISMS.get("gaussian")().check(COUNT, 1.0, 0.0)  # (ε, δ)-DP needs δ > 0


def test_polar_laplace_radius_and_direction():
    task = NumericTask(np.zeros(3), 3.0, 2.0)
    noise = np.array(release("polar_laplace", task, epsilon=0.5, n=8000))
    radius = np.linalg.norm(noise, axis=1)
    assert radius.mean() == pytest.approx(3 * 2.0 / 0.5, rel=0.03)  # Gamma(k, Δ₂/ε): mean kΔ₂/ε
    directions = noise / radius[:, None]
    assert np.abs(directions.mean(axis=0)).max() < 0.03  # uniform on the sphere
    one = np.ravel(release("polar_laplace", NumericTask([0.0], 1.0, 1.0), epsilon=1.0, n=20000))
    assert one.std() == pytest.approx(math.sqrt(2), rel=0.04)  # in dimension 1, the Laplace mechanism


def test_exponential_mechanism_probabilities():
    task = SelectionTask(np.array([10.0, 8.0, 5.0]), 1.0, ["a", "b", "c"], monotone=True)
    picks = np.array(release("exponential", task, epsilon=0.5, n=30000))
    expected = np.exp(0.5 * task.scores)  # monotone scores: exp(ε score / Δ)
    expected /= expected.sum()
    assert np.bincount(picks, minlength=3) / len(picks) == pytest.approx(expected, abs=0.01)
    clear = SelectionTask(np.array([50.0, 48.0, 30.0, 10.0]), 1.0, list("abcd"))
    assert set(release("exponential", clear, epsilon=100.0, n=20)) == {0}


def test_exponential_quantile_stays_in_the_domain_and_converges():
    values = np.random.default_rng(0).normal(50, 10, 2000)
    task = QuantileTask(values, 0.0, 100.0, 0.5)
    outputs = np.array(release("exponential", task, epsilon=0.05, n=200))
    assert np.all((outputs >= 0) & (outputs <= 100))
    precise = np.array(release("exponential", task, epsilon=50.0, n=50))
    assert np.median(precise) == pytest.approx(np.median(values), abs=2.0)


def test_range_queries_answered_query_by_query():
    histogram = np.random.default_rng(0).integers(0, 200, 64).astype(float)
    task = WorkloadTask(histogram, np.tril(np.ones((64, 64))))
    noisy = release("laplace", task, epsilon=1.0)[0]
    assert noisy.shape == (64,)
    assert np.abs(noisy - task.answer).mean() < 64 * 4  # Lap(64/ε): one record is in up to 64 prefixes


def test_mechanisms_reject_unsupported_tasks():
    task = SelectionTask(np.array([1.0, 2.0]), 1.0, ["a", "b"])
    with pytest.raises(ValueError):
        QUERY_MECHANISMS.get("laplace")().check(task, 1.0, 0)


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


def tiny_classic(mechanisms=("laplace", "gaussian", "exponential"), queries=None, tables=None):
    return ClassicConfig(
        name="tiny classic",
        tables=tables or [Spec("census_synthetic", {"n": 2000})],
        queries=queries
        or [
            Spec("count"),
            Spec("mean", {"column": "income"}),
            Spec("quantile", {"column": "income"}),
            Spec("mode", {"column": "region"}),
        ],
        mechanisms=[Spec(m) for m in mechanisms],
        epsilons=[0.5, 2.0],
        repetitions=3,
    )


def test_runner_statuses_and_budget():
    result = run_query_benchmark(tiny_classic())
    ok = result.results[result.results["status"] == "ok"]
    assert (result.results["status"] == "ok").all()
    assert (np.isclose(ok["epsilon_spent"], ok["epsilon"])).all()  # basic composition over the query's parts
    assert any("cannot answer" in s for s in result.skipped)  # e.g. Laplace cannot answer the mode
    assert set(result.summary["table"]) == {"census_synthetic"}
    assert set(result.summary["mechanism"]) == {"laplace", "gaussian", "exponential"}


def test_runner_is_reproducible():
    config = tiny_classic(mechanisms=["laplace"])
    first = run_query_benchmark(config).results["estimate"].tolist()
    assert run_query_benchmark(config).results["estimate"].tolist() == first


def test_several_tables_with_generic_queries():
    config = ClassicConfig(
        name="two tables",
        tables=[Spec("census_synthetic", {"n": 1500}), Spec("diabetes")],
        queries=AUTO,
        mechanisms=[Spec("laplace"), Spec("exponential")],
        epsilons=[1.0],
        repetitions=2,
    )
    summary = run_query_benchmark(config).summary
    assert set(summary["table"]) == {"census_synthetic", "diabetes"}
    counts = summary[(summary["query"] == "Count") & (summary["mechanism"] == "laplace")].set_index("table")
    assert counts.loc["diabetes", "records"] == 442
    labels = {s.label for s in generic_queries(TABLES.get("diabetes")().load())}
    assert {"Count", "Mean", "Median", "100 range queries"} <= labels
