import math

import pytest
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

from dpbench.accounting.accountants import gdp_delta, gdp_epsilon, step_epsilon
from dpbench.accounting.base import UnsupportedNoiseError
from dpbench.accounting.calibration import CalibrationError, calibrate_noise
from dpbench.core.registry import ACCOUNTANTS

DELTA = 1e-5


def gaussian_mechanism_epsilon(sigma: float, delta: float) -> float:
    """Exact epsilon of one Gaussian mechanism with sensitivity 1 and noise sigma (Balle & Wang 2018)."""

    def profile(eps):
        return ndtr(-eps * sigma + 1 / (2 * sigma)) - math.exp(eps + log_ndtr(-eps * sigma - 1 / (2 * sigma))) - delta

    return brentq(profile, 0.0, 100.0, xtol=1e-12)


def accountant(name):
    return ACCOUNTANTS.get(name)()


@pytest.mark.parametrize("sigma", [0.8, 1.0, 2.0, 5.0])
def test_accountants_against_the_exact_gaussian_mechanism(sigma):
    exact = gaussian_mechanism_epsilon(sigma, DELTA)
    for name in ("pld", "prv"):
        value = accountant(name).epsilon(sigma, 1.0, 1, DELTA)
        assert exact - 1e-3 <= value <= exact * 1.02 + 0.01, name
    assert accountant("rdp").epsilon(sigma, 1.0, 1, DELTA) >= exact - 1e-9
    assert accountant("gdp").epsilon(sigma, 1.0, 1, DELTA) == pytest.approx(exact, rel=1e-6)


def test_composition_without_sampling_is_one_gaussian_mechanism():
    sigma, steps = 4.0, 16
    exact = gaussian_mechanism_epsilon(sigma / math.sqrt(steps), DELTA)
    for name in ("pld", "prv", "gdp"):
        assert accountant(name).epsilon(sigma, 1.0, steps, DELTA) == pytest.approx(exact, rel=0.02, abs=0.01), name


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_laplace_mechanism_closed_form(scale):
    # Laplace of scale b and L1 sensitivity 1: delta(eps) = 1 - exp((eps - 1/b) / 2) for eps < 1/b.
    exact = 1 / scale + 2 * math.log(1 - DELTA)
    assert accountant("pld").epsilon(scale, 1.0, 1, DELTA, "laplace") == pytest.approx(exact, abs=2e-3)


def test_dpsgd_regime():
    q, sigma, steps = 256 / 60_000, 1.0, 2350
    rdp, prv, pld = (accountant(n).epsilon(sigma, q, steps, DELTA) for n in ("rdp", "prv", "pld"))
    assert prv == pytest.approx(pld, rel=0.03)
    assert rdp >= max(prv, pld)
    assert 0.9 < pld < 1.5


def test_abadi_et_al_setting():
    # q = 0.01, sigma = 4, 10,000 steps: the moments accountant of Abadi et al. (2016) gives about 1.26;
    # the improved RDP conversion and the numerical accountants are tighter.
    q, sigma, steps = 0.01, 4.0, 10_000
    assert accountant("rdp").epsilon(sigma, q, steps, DELTA) < 1.26
    assert 0.9 < accountant("pld").epsilon(sigma, q, steps, DELTA) < 1.0


@pytest.mark.parametrize("name", ["basic", "advanced", "rdp", "prv", "pld", "gdp"])
def test_monotonicity(name):
    acc = accountant(name)
    q = 0.02
    assert acc.epsilon(2.0, q, 500, DELTA) < acc.epsilon(1.0, q, 500, DELTA)
    assert acc.epsilon(1.0, q, 1000, DELTA) > acc.epsilon(1.0, q, 100, DELTA)
    assert acc.epsilon(1.0, q, 500, 1e-3) < acc.epsilon(1.0, q, 500, 1e-7)


def test_degenerate_cases():
    acc = accountant("rdp")
    assert acc.epsilon(1.0, 0.1, 0, DELTA) == 0.0
    assert math.isinf(acc.epsilon(0.0, 0.1, 10, DELTA))
    with pytest.raises(ValueError):
        acc.epsilon(1.0, 0.0, 10, DELTA)
    with pytest.raises(UnsupportedNoiseError):
        acc.epsilon(1.0, 0.1, 10, DELTA, "laplace")


def test_gdp_formulas_are_inverse():
    for mu in (0.1, 0.5, 1.0, 3.0):
        assert gdp_delta(gdp_epsilon(mu, DELTA), mu) == pytest.approx(DELTA, rel=1e-6)


@pytest.mark.parametrize("name", ["rdp", "prv", "pld"])
@pytest.mark.parametrize("target", [0.5, 2.0, 8.0])
def test_calibrated_noise_is_the_smallest_sufficient(name, target):
    acc = accountant(name)
    q, steps = 0.01, 500
    sigma = calibrate_noise(acc, target, q, steps, DELTA)
    assert acc.epsilon(sigma, q, steps, DELTA) <= target
    assert acc.epsilon(sigma * (1 - 3e-3), q, steps, DELTA) > target


def test_calibration_of_laplace_noise():
    acc = accountant("pld")
    sigma = calibrate_noise(acc, 1.0, 0.01, 500, DELTA, "laplace")
    assert acc.epsilon(sigma, 0.01, 500, DELTA, "laplace") <= 1.0


def test_calibration_edge_cases():
    acc = accountant("rdp")
    assert calibrate_noise(acc, math.inf, 0.01, 100, DELTA) == 0.0
    with pytest.raises(CalibrationError):
        calibrate_noise(acc, 1e-4, 1.0, 1000, DELTA, max_noise=10.0)


def test_composition_theorems_are_looser_than_rdp_and_pld():
    q, sigma, steps = 256 / 60_000, 1.0, 2350
    basic, advanced, rdp, pld = (
        accountant(n).epsilon(sigma, q, steps, DELTA) for n in ("basic", "advanced", "rdp", "pld")
    )
    assert basic >= advanced >= rdp >= pld
    assert basic > 100 * pld  # summing the epsilons of 2,350 steps is hopeless
    assert basic == pytest.approx(steps * step_epsilon(sigma, q, DELTA / steps))


def test_advanced_composition_formula():
    q, sigma, steps = 0.05, 4.0, 400
    eps0 = step_epsilon(sigma, q, DELTA / (2 * steps))
    expected = eps0 * math.sqrt(2 * steps * math.log(2 / DELTA)) + steps * eps0 * math.expm1(eps0)
    assert accountant("advanced").epsilon(sigma, q, steps, DELTA) == pytest.approx(
        min(expected, accountant("basic").epsilon(sigma, q, steps, DELTA))
    )


def test_one_step_without_sampling_is_the_mechanism_itself():
    # q = 1, one step: basic composition is the exact epsilon of one Gaussian (or Laplace) mechanism.
    assert accountant("basic").epsilon(2.0, 1.0, 1, DELTA) == pytest.approx(
        gaussian_mechanism_epsilon(2.0, DELTA), rel=1e-2
    )
    assert accountant("basic").epsilon(0.5, 1.0, 1, DELTA, noise="laplace") == pytest.approx(2.0, rel=1e-2)


def test_tiny_noise_is_fast_and_still_an_upper_bound():
    q, steps = 64 / 1347, 440
    assert accountant("pld").epsilon(0.01, q, steps, DELTA) == pytest.approx(
        accountant("rdp").epsilon(0.01, q, steps, DELTA)
    )
    assert accountant("basic").epsilon(0.01, q, steps, DELTA, noise="laplace") == pytest.approx(
        steps * math.log1p(q * math.expm1(100.0))
    )
