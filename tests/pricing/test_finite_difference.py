"""Accuracy tests for the Crank-Nicolson finite-difference solver against closed-form Black-Scholes."""

import pytest

from dpre.pricing.black_scholes import call_price, put_price
from dpre.pricing.finite_difference import price_crank_nicolson

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.03, 0.2, 0.01
TOLERANCE = 5e-3


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K_", [80.0, 100.0, 120.0])
def test_fd_matches_bs(option_type: str, K_: float) -> None:
    """At a fixed grid resolution, CN price matches closed-form BS within a documented tolerance."""
    bs = call_price(S, K_, T, r, sigma, q) if option_type == "call" else put_price(S, K_, T, r, sigma, q)
    fd = price_crank_nicolson(S, K_, T, r, sigma, option_type, q)
    assert fd == pytest.approx(bs, abs=TOLERANCE)


def test_invalid_option_type_raises() -> None:
    """price_crank_nicolson rejects anything other than 'call'/'put'."""
    with pytest.raises(ValueError):
        price_crank_nicolson(S, K, T, r, sigma, "straddle")
