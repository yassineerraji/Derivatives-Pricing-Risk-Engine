"""Cross-check each closed-form Greek against a high-precision numerical derivative of the BS price itself."""

import pytest

from dpre.greeks.analytical import delta, gamma, rho, theta, vega
from dpre.pricing.black_scholes import price as bs_price

S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.03, 0.22, 0.015
OPTION_TYPES = ["call", "put"]


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_delta_matches_numerical_derivative(option_type: str) -> None:
    """delta ~= d(price)/dS via a tight central difference."""
    h = 1e-4
    numerical = (bs_price(S + h, K, T, r, sigma, option_type, q) - bs_price(S - h, K, T, r, sigma, option_type, q)) / (2 * h)
    assert delta(S, K, T, r, sigma, option_type, q) == pytest.approx(numerical, abs=1e-6)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_gamma_matches_numerical_derivative(option_type: str) -> None:
    """gamma ~= d^2(price)/dS^2 via a central second difference."""
    h = 1e-2
    up = bs_price(S + h, K, T, r, sigma, option_type, q)
    mid = bs_price(S, K, T, r, sigma, option_type, q)
    down = bs_price(S - h, K, T, r, sigma, option_type, q)
    numerical = (up - 2 * mid + down) / h**2
    assert gamma(S, K, T, r, sigma, q) == pytest.approx(numerical, abs=1e-4)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_vega_matches_numerical_derivative(option_type: str) -> None:
    """vega ~= d(price)/d(sigma) via a tight central difference."""
    h = 1e-5
    numerical = (bs_price(S, K, T, r, sigma + h, option_type, q) - bs_price(S, K, T, r, sigma - h, option_type, q)) / (2 * h)
    assert vega(S, K, T, r, sigma, q) == pytest.approx(numerical, abs=1e-5)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_theta_matches_numerical_derivative(option_type: str) -> None:
    """theta = d(price)/dt = -d(price)/dT via a tight central difference in T."""
    h = 1e-5
    dprice_dT = (bs_price(S, K, T + h, r, sigma, option_type, q) - bs_price(S, K, T - h, r, sigma, option_type, q)) / (2 * h)
    assert theta(S, K, T, r, sigma, option_type, q) == pytest.approx(-dprice_dT, abs=1e-4)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_rho_matches_numerical_derivative(option_type: str) -> None:
    """rho ~= d(price)/dr via a tight central difference."""
    h = 1e-5
    numerical = (bs_price(S, K, T, r + h, sigma, option_type, q) - bs_price(S, K, T, r - h, sigma, option_type, q)) / (2 * h)
    assert rho(S, K, T, r, sigma, option_type, q) == pytest.approx(numerical, abs=1e-4)


def test_put_call_parity_on_delta() -> None:
    """delta_call - delta_put == exp(-qT), from differentiating C - P = S*exp(-qT) - K*exp(-rT) w.r.t. S."""
    d_call = delta(S, K, T, r, sigma, "call", q)
    d_put = delta(S, K, T, r, sigma, "put", q)
    import numpy as np

    assert d_call - d_put == pytest.approx(np.exp(-q * T), abs=1e-10)


@pytest.mark.parametrize("fn", [delta, theta, rho])
def test_invalid_option_type_raises(fn) -> None:
    """Every option_type-dependent Greek rejects anything other than 'call'/'put'."""
    with pytest.raises(ValueError):
        fn(S, K, T, r, sigma, "straddle", q)
