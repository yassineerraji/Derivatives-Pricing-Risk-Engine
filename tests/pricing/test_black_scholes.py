"""Correctness tests for closed-form Black-Scholes pricing."""

import numpy as np
import pytest

from dpre.pricing.black_scholes import call_price, put_price

STRIKES = [80.0, 100.0, 120.0]
MATURITIES = [0.1, 1.0, 2.0]
RATES = [0.0, 0.03]
DIV_YIELDS = [0.0, 0.02]
SPOT = 100.0
SIGMA = 0.2


@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("T", MATURITIES)
@pytest.mark.parametrize("r", RATES)
@pytest.mark.parametrize("q", DIV_YIELDS)
def test_put_call_parity(K: float, T: float, r: float, q: float) -> None:
    """C - P == S*exp(-qT) - K*exp(-rT) for every strike/maturity/rate/dividend combination."""
    c = call_price(SPOT, K, T, r, SIGMA, q)
    p = put_price(SPOT, K, T, r, SIGMA, q)
    expected = SPOT * np.exp(-q * T) - K * np.exp(-r * T)
    assert c - p == pytest.approx(expected, abs=1e-10)


def test_call_price_known_value() -> None:
    """Regression check against a widely-cited textbook value (Hull): S=42,K=40,r=0.1,sigma=0.2,T=0.5 -> call ~= 4.759."""
    c = call_price(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)
    assert c == pytest.approx(4.759, abs=1e-3)


def test_deep_itm_call_converges_to_intrinsic() -> None:
    """A deep in-the-money call with near-zero vol should price close to its (discounted) intrinsic value."""
    c = call_price(S=200.0, K=100.0, T=1.0, r=0.0, sigma=1e-6)
    assert c == pytest.approx(100.0, abs=1e-3)


def test_invalid_option_type_raises() -> None:
    """price() dispatch rejects anything other than 'call'/'put'."""
    from dpre.pricing.black_scholes import price

    with pytest.raises(ValueError):
        price(100.0, 100.0, 1.0, 0.0, 0.2, "straddle")
