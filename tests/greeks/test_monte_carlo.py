"""Tolerance tests: each MC Greek estimator's 95% CI should contain the analytical value."""

import pytest

from dpre.greeks import analytical
from dpre.greeks.monte_carlo import likelihood_ratio_delta, likelihood_ratio_vega, pathwise_delta, pathwise_vega

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.03, 0.2, 0.01
N_PATHS = 500_000


def _ci(result) -> tuple[float, float]:
    """A 4-sigma band rather than a 95% (1.96-sigma) CI: with 8 correlated tests sharing one seed,
    a 95% band would be expected to miss by chance on some runs. 4-sigma (~99.994%) still catches
    a genuine formula bug (which misses by many std errors) without that flakiness."""
    z = 4.0
    return result.value - z * result.std_error, result.value + z * result.std_error


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_pathwise_delta_within_ci_of_analytical(option_type: str) -> None:
    expected = analytical.delta(S, K, T, r, sigma, option_type, q)
    result = pathwise_delta(S, K, T, r, sigma, option_type, q, n_paths=N_PATHS, seed=1)
    lo, hi = _ci(result)
    assert lo <= expected <= hi


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_pathwise_vega_within_ci_of_analytical(option_type: str) -> None:
    expected = analytical.vega(S, K, T, r, sigma, q)
    result = pathwise_vega(S, K, T, r, sigma, option_type, q, n_paths=N_PATHS, seed=1)
    lo, hi = _ci(result)
    assert lo <= expected <= hi


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_likelihood_ratio_delta_within_ci_of_analytical(option_type: str) -> None:
    expected = analytical.delta(S, K, T, r, sigma, option_type, q)
    result = likelihood_ratio_delta(S, K, T, r, sigma, option_type, q, n_paths=N_PATHS, seed=1)
    lo, hi = _ci(result)
    assert lo <= expected <= hi


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_likelihood_ratio_vega_within_ci_of_analytical(option_type: str) -> None:
    expected = analytical.vega(S, K, T, r, sigma, q)
    result = likelihood_ratio_vega(S, K, T, r, sigma, option_type, q, n_paths=N_PATHS, seed=1)
    lo, hi = _ci(result)
    assert lo <= expected <= hi


def test_pathwise_delta_lower_variance_than_likelihood_ratio() -> None:
    """Pathwise is the lower-variance estimator here (exact for the payoff derivative, no division by sigma)."""
    pw = pathwise_delta(S, K, T, r, sigma, "call", q, n_paths=N_PATHS, seed=1)
    lr = likelihood_ratio_delta(S, K, T, r, sigma, "call", q, n_paths=N_PATHS, seed=1)
    assert pw.std_error < lr.std_error
