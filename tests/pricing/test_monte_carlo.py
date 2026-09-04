"""Convergence and variance-reduction tests for the Monte Carlo pricer."""

import pytest

from dpre.pricing.black_scholes import call_price, put_price
from dpre.pricing.monte_carlo import mc_price

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.03, 0.2, 0.01


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("control_variate", [True, False])
def test_mc_price_within_ci_of_bs(option_type: str, antithetic: bool, control_variate: bool) -> None:
    """A large-sample MC estimate's 95% CI should contain the closed-form Black-Scholes price."""
    bs = call_price(S, K, T, r, sigma, q) if option_type == "call" else put_price(S, K, T, r, sigma, q)
    result = mc_price(
        S, K, T, r, sigma, option_type, q,
        n_paths=200_000, seed=7, antithetic=antithetic, control_variate=control_variate,
    )
    assert result.ci_low <= bs <= result.ci_high


def test_control_variate_reduces_variance() -> None:
    """Control variate should lower estimator variance for a call (positive correlation with S_T)."""
    raw = mc_price(S, K, T, r, sigma, "call", q, n_paths=50_000, seed=1, antithetic=False, control_variate=False)
    cv = mc_price(S, K, T, r, sigma, "call", q, n_paths=50_000, seed=1, antithetic=False, control_variate=True)
    assert cv.variance < raw.variance


def test_antithetic_reduces_variance() -> None:
    """Antithetic variates should lower estimator variance for a monotonic call payoff."""
    raw = mc_price(S, K, T, r, sigma, "call", q, n_paths=50_000, seed=1, antithetic=False, control_variate=False)
    anti = mc_price(S, K, T, r, sigma, "call", q, n_paths=50_000, seed=1, antithetic=True, control_variate=False)
    assert anti.variance < raw.variance


def test_error_shrinks_with_more_paths() -> None:
    """Standard error should shrink as the number of simulated paths grows."""
    small = mc_price(S, K, T, r, sigma, "call", q, n_paths=1_000, seed=3)
    large = mc_price(S, K, T, r, sigma, "call", q, n_paths=100_000, seed=3)
    assert large.std_error < small.std_error
