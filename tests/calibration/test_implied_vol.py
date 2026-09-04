"""Round-trip tests for the IV solvers and the chain-level extraction wrapper."""

from datetime import date

import pandas as pd
import pytest

from dpre.calibration.data import MarketSnapshot
from dpre.calibration.implied_vol import extract_iv_surface, implied_vol_brent, implied_vol_newton
from dpre.pricing.black_scholes import price as bs_price

S, T, r, q = 100.0, 0.75, 0.03, 0.01
TRUE_VOLS = [0.10, 0.20, 0.35, 0.60]
STRIKES = [80.0, 100.0, 120.0]


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("true_vol", TRUE_VOLS)
def test_brent_recovers_known_vol(true_vol: float, K: float, option_type: str) -> None:
    """Brent solver recovers the vol used to generate the market price."""
    market_price = bs_price(S, K, T, r, true_vol, option_type, q)
    recovered = implied_vol_brent(market_price, S, K, T, r, option_type, q)
    assert recovered == pytest.approx(true_vol, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("true_vol", TRUE_VOLS)
def test_newton_recovers_known_vol(true_vol: float, K: float, option_type: str) -> None:
    """Newton-Raphson solver recovers the vol used to generate the market price."""
    market_price = bs_price(S, K, T, r, true_vol, option_type, q)
    recovered = implied_vol_newton(market_price, S, K, T, r, option_type, q)
    assert recovered == pytest.approx(true_vol, abs=1e-6)


def test_brent_rejects_unbracketed_price() -> None:
    """A price above the max achievable value (vol=hi) has no bracketed root and should raise."""
    absurd_price = S * 10
    with pytest.raises(ValueError):
        implied_vol_brent(absurd_price, S, 100.0, T, r, "call", q)


def test_extract_iv_surface_recovers_vols_and_drops_bad_quotes() -> None:
    """extract_iv_surface solves per-row IV from mid price and drops quotes with no valid root."""
    true_vol = 0.25
    good_mid = bs_price(S, 100.0, T, r, true_vol, "call", q)
    chain = pd.DataFrame(
        {
            "strike": [100.0, 100.0],
            "bid": [good_mid - 0.01, S * 20],
            "ask": [good_mid + 0.01, S * 20 + 0.01],
            "mid": [good_mid, S * 20],
            "T": [T, T],
            "option_type": ["call", "call"],
            "expiry": [date(2027, 1, 1), date(2027, 1, 1)],
            "volume": [10, 10],
            "open_interest": [100, 100],
        }
    )
    snapshot = MarketSnapshot(ticker="TEST", as_of=date.today(), spot=S, risk_free_rate=r, dividend_yield=q, chain=chain)

    with pytest.warns(UserWarning):
        iv_df = extract_iv_surface(snapshot)

    assert len(iv_df) == 1
    assert iv_df["implied_vol"].iloc[0] == pytest.approx(true_vol, abs=1e-6)
