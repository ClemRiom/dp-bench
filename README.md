# DP-Bench

A benchmark of differentially private algorithms, driven from a dashboard. The dashboard
opens on an empty page: every comparison is built from the interface.

```bash
pip install -e ".[dashboard]"
dpbench dashboard
```

## Machine learning

* **Training algorithms**: non-private SGD, DP-SGD, DP-SGD with Opacus, DP-Adam, DP-LAP-SGD
  (Laplace noise) and GeoDP (radial + directional noise, Duan et al., ICDE 2025, with its
  certified worst-case epsilon).
* **Accountants**, chosen in the form, several at once to compare them: basic and advanced
  composition, RDP, PRV, PLD and GDP. The noise of every training is calibrated so that the
  chosen accountant certifies the target epsilon.
* **Datasets**: digits, synthetic, Adult, MNIST, Fashion-MNIST, CIFAR-10 (downloaded once), or
  your own file. **Models**: mlp, cnn, cnn_gn.
* **Metrics**: accuracy, macro F1, ROC AUC, cross-entropy, and three fairness gaps (accuracy
  between groups, demographic parity, accuracy between classes).

Pages: *Run an experiment*, *Results* (utility, epsilon spent, fairness, tables) and
*Accountants* (epsilon of a training according to each accountant, noise needed for a target).

## Classic DP on databases

Nine kinds of queries (count, sum, mean, histogram, 2-way marginal, centroid, range queries,
quantile, mode) and thirteen mechanisms: Laplace, Gaussian (analytic and classic
calibration), discrete Laplace, polar Laplace, exponential mechanism, report noisy max,
permute-and-flip, noisy histogram quantile, identity, uniform, hierarchical tree and MWEM. One
database, or several compared with the same types of queries.

## Adding your own

Datasets, training algorithms, query mechanisms (a class, or a single function) and
accountants can be added from the *Add a dataset* and *Add an algorithm* pages, without
changing DP-Bench; they are saved in `workspace/`. The API is described in
[EXTENDING.md](EXTENDING.md).

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow and not network"
```
