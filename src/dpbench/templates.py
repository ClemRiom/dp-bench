"""Starting points shown in the dashboard's "Add an algorithm" page.

Each template is a complete, working plugin: it can be saved as it is, then modified.
"""

TRAINING_TEMPLATE = '''"""DP-SGD with automatic clipping (Bu et al., NeurIPS 2023): a DP-Bench plugin.

DP-SGD clips each per-example gradient g to norm C: g * min(1, C / ‖g‖).
Automatic clipping rescales it instead to C * g / (‖g‖ + γ). Every rescaled gradient
still has a norm below C, so the sensitivity of the sum of the batch is still C, and
the privacy accounting of DP-SGD applies unchanged: only the clipping rule changes.

To write your own variant, override a method of the engine (PrivateGradient):
  clip_factors(norms)          the factor applied to each per-example gradient
  sample_noise(like)           the noise added to the clipped sum
  privatize(x, y, batch_size)  the whole private gradient of a batch (see geodp.py)
or change class attributes of DPSGD (noise, clipping_norm, default_optimizer...).
"""

from dpbench.core.registry import register_mechanism
from dpbench.mechanisms.dpsgd import DPSGD
from dpbench.mechanisms.engine import PrivateGradient

GAMMA = 0.01  # stability constant γ of automatic clipping


class AutoClipGradient(PrivateGradient):
    """Private gradient of DP-SGD with automatic clipping."""

    def clip_factors(self, norms):
        # Factor applied to each per-example gradient: C / (‖g‖ + γ) instead of min(1, C / ‖g‖).
        return self.max_grad_norm / (norms + GAMMA)


@register_mechanism("dpsgd_autoclip")
class DPSGDAutoClip(DPSGD):
    """DP-SGD with automatic clipping (Bu et al. 2023): per-example gradients rescaled to C·g/(‖g‖+γ)."""

    title = "DP-SGD (auto-clip)"
    engine_class = AutoClipGradient
    default_lr = 0.5
'''

QUERY_TEMPLATE = '''"""Laplace mechanism with post-processing of the counts: a DP-Bench query mechanism.

release() receives a task and must return its private answer:
  task.kind == "numeric": task.answer (a vector), task.l1_sensitivity, task.l2_sensitivity, task.integer
  task.kind == "workload": task.histogram, task.workload (answer = workload @ histogram)
numeric_view(task) gives (answer, L1 sensitivity, L2 sensitivity, integer) for both kinds.

Rounding the noisy counts and replacing negative values by 0 only uses the noisy answer,
not the data: it is post-processing, so the mechanism stays epsilon-DP.
"""

import numpy as np

from dpbench.classic.mechanisms import QueryMechanism, numeric_view
from dpbench.core.registry import register_query_mechanism


@register_query_mechanism("laplace_postprocessed")
class LaplacePostprocessed(QueryMechanism):
    """Laplace mechanism, then counts rounded and clipped at 0 (post-processing). Pure ε-DP."""

    title = "Laplace + post-processing"
    supports = ("numeric", "workload")  # the kinds of tasks this mechanism can release
    uses_delta = False  # pure epsilon-DP

    def release(self, task, epsilon, delta, rng):
        answer, l1_sensitivity, _, integer = numeric_view(task)
        noisy = answer + rng.laplace(0.0, l1_sensitivity / epsilon, answer.shape)
        if integer:  # counts: integers, never negative
            noisy = np.clip(np.round(noisy), 0, None)
        return noisy
'''

FUNCTION_TEMPLATE = '''"""The Laplace mechanism written as one function: a DP-Bench query mechanism.

@noise_mechanism turns the function into a mechanism. The function receives:
  answer       the exact answer, a vector of floats (a count is a vector of length 1)
  sensitivity  how much one record can change the answer (L1 norm; sensitivity="l2" for L2)
  epsilon      the budget of this release (delta: only for (epsilon, delta)-DP mechanisms)
  rng          a NumPy random generator: draw all the noise from it, so runs are reproducible
and returns the noisy answer, with the same shape. The mechanism then answers every numeric
query (counts, sums, means, histograms, centroids) and range queries answered one by one.
"""

from dpbench.classic import noise_mechanism


@noise_mechanism("my_laplace", title="My Laplace")
def my_laplace(answer, sensitivity, epsilon, delta, rng):
    """Laplace noise of scale sensitivity / epsilon on every value. Pure epsilon-DP."""
    return answer + rng.laplace(0.0, sensitivity / epsilon, size=answer.shape)
'''

ACCOUNTANT_TEMPLATE = '''"""An accountant that ignores the amplification by subsampling: a DP-Bench accountant plugin.

An accountant answers one question: after `steps` steps that each sample every example with
probability `sample_rate` and add noise of multiplier `noise_multiplier` (the noise std
divided by the clipping bound), what is the smallest epsilon such that the whole training is
(epsilon, delta)-DP? Implement _epsilon(); the base class checks the arguments.

This one pretends that every step uses the whole dataset (sample_rate = 1). T Gaussian
steps of multiplier sigma then compose exactly into one Gaussian mechanism with
mu = sqrt(T) / sigma (Gaussian DP, Dong, Roth & Su 2022). Subsampling can only help, so this
is a valid upper bound, but a very loose one: compare it with RDP on the Accountants page.
"""

import math

from dpbench.accounting.accountants import gdp_epsilon
from dpbench.accounting.base import Accountant
from dpbench.core.registry import register_accountant


@register_accountant("no_amplification")
class NoAmplification(Accountant):
    """Gaussian DP composition without amplification by subsampling: mu = sqrt(T) / sigma. Upper bound, loose."""

    supported_noises = ("gaussian",)  # the kinds of noise this accountant can analyse
    approximate = False  # True for an approximation that is not a guarantee

    def _epsilon(self, noise_multiplier, sample_rate, steps, delta, noise):
        return gdp_epsilon(math.sqrt(steps) / noise_multiplier, delta)
'''

DATASET_TEMPLATE = '''"""The Wine recognition dataset of scikit-learn (bundled, no download): a DP-Bench dataset plugin.

load_full() returns a Dataset split into a training set and a test set.
For simplicity, the features are scaled with the minimum and maximum of the whole file:
like in many papers, this preprocessing is not private. Replace these bounds by public
ones (documentation, domain knowledge) when you have them.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from dpbench.core.registry import register_dataset
from dpbench.core.types import Dataset
from dpbench.datasets.base import DatasetLoader


@register_dataset("wine_plugin")
class Wine(DatasetLoader):
    """Wine recognition (scikit-learn): 178 wines, 13 chemical features, 3 cultivars."""

    def load_full(self) -> Dataset:
        data = load_wine()
        low, high = np.floor(data.data.min(axis=0)), np.ceil(data.data.max(axis=0))  # not private: see above
        X = ((data.data - low) / (high - low)).astype(np.float32)
        X_tr, X_te, y_tr, y_te = train_test_split(X, data.target, test_size=0.25, random_state=0, stratify=data.target)
        return Dataset(
            name="wine_plugin",
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            n_classes=3,
            class_names=list(data.target_names),
            feature_names=list(data.feature_names),
            description="Wine recognition (UCI), 178 records",
            source="scikit-learn load_wine",
        )
'''
