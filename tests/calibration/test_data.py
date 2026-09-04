"""Tests for the pure (network-free) chain-cleaning transformation."""

from datetime import date

import pandas as pd

from dpre.calibration.data import _clean_quotes, _select_expiries


def test_clean_quotes_uses_bid_ask_mid_when_live_quote_exists() -> None:
    """A row with a live bid/ask gets its midpoint and is tagged quote_source='bid_ask'."""
    raw = pd.DataFrame(
        {"strike": [90.0], "bid": [5.0], "ask": [5.2], "lastPrice": [5.05], "volume": [10], "openInterest": [100]}
    )
    cleaned = _clean_quotes(raw, expiry=date(2026, 12, 18), T=0.5, option_type="call")

    assert cleaned["mid"].iloc[0] == 5.1
    assert cleaned["quote_source"].iloc[0] == "bid_ask"
    assert cleaned["expiry"].iloc[0] == date(2026, 12, 18)
    assert cleaned["T"].iloc[0] == 0.5
    assert cleaned["option_type"].iloc[0] == "call"
    assert "open_interest" in cleaned.columns


def test_clean_quotes_falls_back_to_last_price_when_traded_today() -> None:
    """No live bid/ask but volume > 0 today: use lastPrice, tagged quote_source='last_price'."""
    raw = pd.DataFrame(
        {"strike": [100.0], "bid": [0.0], "ask": [0.0], "lastPrice": [9.0], "volume": [500], "openInterest": [0]}
    )
    cleaned = _clean_quotes(raw, expiry=date(2026, 12, 18), T=0.5, option_type="call")

    assert cleaned["mid"].iloc[0] == 9.0
    assert cleaned["quote_source"].iloc[0] == "last_price"


def test_clean_quotes_drops_untraded_unquoted_rows() -> None:
    """No live bid/ask and no volume today: no reliable price, row is dropped rather than fabricated."""
    raw = pd.DataFrame(
        {"strike": [110.0], "bid": [0.0], "ask": [0.0], "lastPrice": [2.0], "volume": [0], "openInterest": [50]}
    )
    cleaned = _clean_quotes(raw, expiry=date(2026, 12, 18), T=0.5, option_type="call")

    assert cleaned.empty


def test_select_expiries_spreads_across_horizon_for_dense_listings() -> None:
    """With more daily-listed expiries than max_expiries, selection spans the full horizon, not just the front."""
    as_of = date(2026, 1, 1)
    daily = [(as_of + pd.Timedelta(days=d)).isoformat() for d in range(1, 400)]

    selected = _select_expiries(daily, as_of, max_expiries=6, max_horizon_days=365)

    assert len(selected) == 6
    days_out = [(date.fromisoformat(e) - as_of).days for e in selected]
    assert days_out == sorted(days_out)
    assert days_out[0] < 30
    assert days_out[-1] > 300


def test_select_expiries_returns_all_when_fewer_than_requested() -> None:
    """When there are fewer future expiries than max_expiries, all of them are kept."""
    as_of = date(2026, 1, 1)
    sparse = ["2026-02-01", "2026-06-01"]

    selected = _select_expiries(sparse, as_of, max_expiries=6)

    assert selected == sparse


def test_select_expiries_excludes_zero_dte() -> None:
    """Same-day (0DTE) expiries are excluded since T=0 is not usable for calibration."""
    as_of = date(2026, 1, 1)
    selected = _select_expiries(["2026-01-01", "2026-02-01"], as_of, max_expiries=6)

    assert "2026-01-01" not in selected
