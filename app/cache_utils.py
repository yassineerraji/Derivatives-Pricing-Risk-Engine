"""Cached wrappers around the dpre pipeline so widget interactions don't re-fetch or re-calibrate."""

import numpy as np
import streamlit as st
import yfinance as yf

from dpre.calibration.data import fetch_chain
from dpre.calibration.implied_vol import extract_iv_surface
from dpre.calibration.svi import calibrate_svi_surface
from dpre.risk.var_es import fetch_historical_log_returns

DEFAULT_TICKER = "SPY"
DEFAULT_MAX_EXPIRIES = 8


@st.cache_data(ttl=3600, show_spinner="Fetching option chain and calibrating the SVI surface...")
def get_calibration(ticker: str, dividend_yield: float, max_expiries: int = DEFAULT_MAX_EXPIRIES):
    """Fetch the chain, extract implied vol, and calibrate an arbitrage-constrained SVI surface.

    Cached for an hour, keyed on (ticker, dividend_yield, max_expiries): this is the same live
    pipeline as scripts/02_calibrate_surface.py, just reused across every page and widget
    interaction instead of re-run on each one, and re-run automatically when either the ticker or
    the dividend-yield override actually changes.
    """
    snapshot = fetch_chain(ticker, max_expiries=max_expiries, dividend_yield=dividend_yield)
    iv_df = extract_iv_surface(snapshot)
    svi_slices = calibrate_svi_surface(iv_df, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield)
    return snapshot, iv_df, svi_slices


@st.cache_data(ttl=3600, show_spinner="Fetching historical returns...")
def get_historical_returns(ticker: str = DEFAULT_TICKER, lookback_days: int = 250) -> np.ndarray:
    """Cached wrapper around risk.var_es.fetch_historical_log_returns."""
    return fetch_historical_log_returns(ticker, lookback_days)


@st.cache_data(ttl=86400, show_spinner=False)
def suggest_dividend_yield(ticker: str) -> float:
    """Best-effort trailing dividend yield for `ticker`, as a fraction (not a percent).

    Falls back to 0.0 on any failure -- a safer default across arbitrary tickers than the SPY-specific
    constant this project otherwise uses (most single names yield less than SPY, many yield nothing),
    and this is always shown to the user as an editable, overridable value rather than applied silently.

    yfinance's `dividendYield` field is a raw percentage-point number (SPY -> 1.01, meaning 1.01%,
    not 1.01 already-a-fraction), verified directly against live data; `trailingAnnualDividendYield`
    is already a fraction. These need different conversions, not a shared "divide by 100 if > 1.0"
    guess -- that heuristic looks right for SPY (yield just above 1%) and is wrong for anything
    under 1% (e.g. AAPL ~0.33 for 0.33% got read as 33% and clamped to the ceiling).
    """
    try:
        info = yf.Ticker(ticker).get_info()
        if info.get("dividendYield") is not None:
            y = float(info["dividendYield"]) / 100.0
        elif info.get("trailingAnnualDividendYield") is not None:
            y = float(info["trailingAnnualDividendYield"])
        else:
            y = 0.0
        return max(0.0, min(y, 0.15))
    except Exception:
        return 0.0
