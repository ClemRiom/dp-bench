# DP-Bench, step 2: classic DP on databases

```bash
pip install -e ".[dashboard]"
dpbench dashboard
```

One page, **Queries on databases**: choose a table, the queries (9 types) and the mechanisms
(Laplace, Gaussian, polar Laplace, exponential), then run; or compare several databases with
the same types of queries.

Tests: `pip install -e ".[dev]"` then `pytest`.
