# DP-Bench, end of month 2 (version 0.3.0)

Added since the simplified sprint-2 version:

* 13 query mechanisms instead of 4: classic Gaussian, discrete Laplace, report noisy max,
  permute-and-flip, noisy histogram quantile, identity, uniform, hierarchical and MWEM are back.
* `@noise_mechanism`: a query mechanism written as one function (template on *Add an algorithm*).
* Accountants: choice in the *Run an experiment* form (several at once to compare them), two new
  ones (basic and advanced composition), the *Accountants* page, an accountant template, and the
  accountant used shown in the calibration table of *Results*.
* An interface without icons or captions.
* EXTENDING.md: the API of every kind of plugin.

Still left out (kept in the full sprint-2 version): YAML experiments, reference results, the HTML
report, the Mechanism focus and Plugins pages, the logreg model, membership inference attacks.
