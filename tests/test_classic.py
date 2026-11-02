"""Classic DP: schemas, queries (exact answers and sensitivities), mechanisms (noise distributions), runner."""

import math

import numpy as np
import pandas as pd
import pytest

from dpbench.accounting.accountants import gdp_delta
from dpbench.classic.mechanisms import analytic_gaussian_sigma, classic_gaussian_sigma, noise_mechanism
from dpbench.classic.runner import AUTO, ClassicConfig, Spec, generic_queries, run_query_benchmark
from dpbench.classic.tables import Table
from dpbench.classic.tasks import NumericTask, QuantileTask, SelectionTask, WorkloadTask
from dpbench.core.registry import QUERIES, QUERY_MECHANISMS, TABLES
from dpbench.core.schema import CATEGORICAL, IGNORED, NUMERIC, ColumnSpec, TableSchema, infer_schema


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
    schema = infer_schema(frame)
    schema.columns["age"].lower, schema.columns["age"].upper = 0.0, 100.0
    schema.columns["income"].lower, schema.columns["income"].upper = 0.0, 100.0
    return Table("toy", frame, schema)


def release(name, task, epsilon=1.0, delta=1e-6, n=1, seed=0, **params):
    mechanism = QUERY_MECHANISMS.get(name)(**params)
    rng = np.random.default_rng(seed)
    return [mechanism.release(task, epsilon, delta, rng) for _ in range(n)]


# --------------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------------


def test_infer_schema_types_and_public_bounds():
    codes = pd.DataFrame({"level": [1, 2, 3] * 40})
    assert infer_schema(codes).columns["level"].kind == CATEGORICAL  # a few integer codes: categories
    frame = pd.DataFrame(
        {"age": [23, 41, 67], "city": ["a", "b", "a"], "id": [f"x{i}" for i in range(3)], "w": [0.5, 1.5, 2.5]}
    )
    schema = infer_schema(frame)
    assert schema.columns["age"].kind == NUMERIC and schema.columns["city"].kind == CATEGORICAL
    assert schema.columns["age"].lower <= 23 and schema.columns["age"].upper >= 67  # rounded outwards
    assert schema.columns["city"].categories == ["a", "b"]
    assert schema.columns["id"].kind in (CATEGORICAL, IGNORED)
    assert set(schema.bounds_from_data) >= {"age", "city", "w"}  # flagged: computed from the data
    assert type(schema).from_dict(schema.to_dict()).to_dict() == schema.to_dict()


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


def test_thirteen_query_mechanisms():
    assert set(QUERY_MECHANISMS.names()) >= {
        "laplace", "gaussian", "gaussian_classic", "discrete_laplace", "polar_laplace", "exponential",
        "report_noisy_max", "permute_and_flip", "histogram_quantile", "identity", "uniform", "hierarchical", "mwem",
    }  # fmt: skip


def test_laplace_scale():
    noise = np.ravel(release("laplace", COUNT, epsilon=0.5, n=20000)) - 1000
    assert abs(noise.mean()) < 0.1
    assert noise.std() == pytest.approx(math.sqrt(2) * 2.0, rel=0.04)  # Lap(Δ/ε), b = 2


def test_gaussian_calibrations():
    assert classic_gaussian_sigma(0.5, 1e-5, 1.0) == pytest.approx(math.sqrt(2 * math.log(1.25 / 1e-5)) / 0.5)
    with pytest.raises(ValueError):
        classic_gaussian_sigma(1.0, 1e-5, 1.0)  # the classic proof needs ε < 1
    for epsilon in (0.1, 0.5, 1.0, 4.0):
        sigma = analytic_gaussian_sigma(epsilon, 1e-6, 1.0)
        assert gdp_delta(epsilon, 1.0 / sigma) == pytest.approx(1e-6, rel=1e-6)  # exact (ε, δ) profile
        if epsilon < 1:
            assert sigma < classic_gaussian_sigma(epsilon, 1e-6, 1.0)  # the analytic σ is always smaller
    noise = np.ravel(release("gaussian", COUNT, epsilon=1.0, n=20000)) - 1000
    assert noise.std() == pytest.approx(analytic_gaussian_sigma(1.0, 1e-6, 1.0), rel=0.03)
    with pytest.raises(ValueError):
        QUERY_MECHANISMS.get("gaussian_classic")().check(COUNT, 1.0, 1e-6)  # skipped by the runner


def test_discrete_laplace_is_integer_with_the_right_distribution():
    outputs = np.ravel(release("discrete_laplace", COUNT, epsilon=1.0, n=20000))
    noise = outputs - 1000
    assert np.all(noise == np.round(noise))
    alpha = math.exp(-1.0)
    assert np.mean(noise == 0) == pytest.approx((1 - alpha) / (1 + alpha), abs=0.015)  # P(0) = (1-α)/(1+α)
    assert noise.var() == pytest.approx(2 * alpha / (1 - alpha) ** 2, rel=0.06)
    with pytest.raises(ValueError):
        QUERY_MECHANISMS.get("discrete_laplace")().check(NumericTask([0.5], 1.0, 1.0), 1.0, 0)


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


def test_selection_mechanisms_find_a_clear_winner_and_pf_is_not_worse_than_em():
    task = SelectionTask(np.array([50.0, 48.0, 30.0, 10.0]), 1.0, list("abcd"))
    for name in ("exponential", "report_noisy_max", "permute_and_flip"):
        assert set(release(name, task, epsilon=100.0, n=20)) == {0}
    em = np.array(release("exponential", task, epsilon=0.3, n=6000, seed=1))
    pf = np.array(release("permute_and_flip", task, epsilon=0.3, n=6000, seed=2))
    regret = lambda picks: float(np.mean(task.scores.max() - task.scores[picks]))  # noqa: E731
    assert regret(pf) <= regret(em) * 1.05  # McKenna & Sheldon: never worse in expectation


def test_quantile_mechanisms_stay_in_the_domain_and_converge():
    values = np.random.default_rng(0).normal(50, 10, 2000)
    task = QuantileTask(values, 0.0, 100.0, 0.5)
    for name in ("exponential", "histogram_quantile"):
        outputs = np.array(release(name, task, epsilon=0.05, n=200))
        assert np.all((outputs >= 0) & (outputs <= 100))
        precise = np.array(release(name, task, epsilon=50.0, n=50))
        assert np.median(precise) == pytest.approx(np.median(values), abs=2.0)


def workload_error(name, task, epsilon=1.0, n=20, **params):
    outputs = release(name, task, epsilon=epsilon, n=n, **params)
    return float(np.mean([np.abs(o - task.answer).mean() for o in outputs]))


def test_workload_mechanisms():
    rng = np.random.default_rng(0)
    histogram = rng.integers(0, 200, 256).astype(float)
    prefixes = WorkloadTask(histogram, np.tril(np.ones((256, 256))))
    assert np.allclose(release("hierarchical", prefixes, epsilon=1e9)[0], prefixes.answer, atol=1e-3)  # unbiased
    uniform = release("uniform", prefixes, epsilon=1e9)[0]
    assert np.allclose(uniform, np.full(256, histogram.mean()).cumsum())
    # Laplace on every answer: one record is in up to 256 prefixes, so the noise is huge.
    assert workload_error("laplace", prefixes) > 5 * workload_error("identity", prefixes)
    # MWEM starts from the uniform histogram and improves it with the measured queries.
    small = WorkloadTask(histogram[:64], np.tril(np.ones((64, 64))))
    uniform_error = np.abs(np.full(64, small.answer[-1] / 64).cumsum() - small.answer).mean()
    assert workload_error("mwem", small, epsilon=10.0, n=5) < uniform_error


def test_hierarchical_beats_identity_on_long_ranges_of_a_large_domain():
    # Hay et al. (2010): a long range sums up to m noisy cells with the identity strategy, but only
    # O(log m) noisy nodes of the tree. With 4096 cells and ranges of 2048 to 4096 cells, the tree wins.
    histogram = np.random.default_rng(1).integers(0, 50, 4096).astype(float)
    lengths = np.linspace(2048, 4096, 32).astype(int)
    workload = (np.arange(4096)[None, :] < lengths[:, None]).astype(float)
    task = WorkloadTask(histogram, workload)
    assert workload_error("hierarchical", task, n=10) < workload_error("identity", task, n=10)


def test_noise_mechanism_turns_a_function_into_a_mechanism():
    @noise_mechanism("test_function_laplace", title="Function Laplace")
    def function_laplace(answer, sensitivity, epsilon, delta, rng):
        return answer + rng.laplace(0.0, sensitivity / epsilon, size=answer.shape)

    try:
        cls = QUERY_MECHANISMS.get("test_function_laplace")
        assert cls.title == "Function Laplace" and cls.supports == ("numeric", "workload") and not cls.uses_delta
        noise = np.ravel(release("test_function_laplace", COUNT, epsilon=0.5, n=20000)) - 1000
        assert noise.std() == pytest.approx(math.sqrt(2) * 2.0, rel=0.04)  # same law as the built-in Laplace
        task = WorkloadTask(np.ones(8), np.tril(np.ones((8, 8))))
        assert release("test_function_laplace", task)[0].shape == (8,)  # range queries answered one by one
        with pytest.raises(ValueError):
            QUERY_MECHANISMS.get("test_function_laplace")().check(SelectionTask([1.0], 1.0, ["a"]), 1.0, 0.0)
    finally:
        QUERY_MECHANISMS.unregister("test_function_laplace")

    @noise_mechanism("test_function_broken")
    def broken(answer, sensitivity, epsilon, delta, rng):
        return answer[:0]

    try:
        with pytest.raises(ValueError, match="shape"):
            release("test_function_broken", COUNT)
    finally:
        QUERY_MECHANISMS.unregister("test_function_broken")
    with pytest.raises(ValueError):
        noise_mechanism("bad", sensitivity="l3")


def test_mechanisms_reject_unsupported_tasks():
    task = SelectionTask(np.array([1.0, 2.0]), 1.0, ["a", "b"])
    with pytest.raises(ValueError):
        QUERY_MECHANISMS.get("laplace")().check(task, 1.0, 0)


# --------------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------------


def tiny_classic(
    mechanisms=("laplace", "gaussian", "gaussian_classic", "discrete_laplace", "exponential"), queries=None, tables=None
):
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
    results = result.results
    ok = results[results["status"] == "ok"]
    assert set(results["status"]) == {"ok", "skipped"}
    assert (np.isclose(ok["epsilon_spent"], ok["epsilon"])).all()  # basic composition over the query's parts
    skipped = results[results["status"] == "skipped"]
    assert set(skipped.loc[skipped["epsilon"] == 2.0, "mechanism"]) >= {"gaussian_classic"}  # classic σ needs ε < 1
    assert "discrete_laplace" in set(skipped.loc[skipped["query"] == "mean", "mechanism"])  # a sum is not an integer
    assert any("cannot answer" in s for s in result.skipped)  # e.g. Laplace cannot answer the mode
    assert set(result.summary["table"]) == {"census_synthetic"}
    assert set(result.summary["mechanism"]) == {
        "laplace",
        "gaussian",
        "gaussian_classic",
        "discrete_laplace",
        "exponential",
    }


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
