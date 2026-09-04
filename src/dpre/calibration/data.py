"""Fetch an options chain and spot/rate/dividend assumptions from yfinance, caching raw pulls locally."""

import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
DEFAULT_TICKER = "SPY"
DEFAULT_DIVIDEND_YIELD = 0.013
DEFAULT_RISK_FREE_RATE = 0.05


@dataclass
class MarketSnapshot:
    """One options-chain pull: spot price, rate, dividend yield, and per-expiry call/put quotes (chain)."""

    ticker: str
    as_of: date
    spot: float
    risk_free_rate: float
    dividend_yield: float
    chain: pd.DataFrame


def get_risk_free_rate() -> float:
    """Best-effort 13-week T-bill yield (^IRX) as a risk-free-rate proxy; falls back to a fixed constant."""
    try:
        history = yf.Ticker("^IRX").history(period="5d")["Close"]
        if history.empty:
            raise ValueError("empty ^IRX history")
        return float(history.iloc[-1]) / 100.0
    except Exception as exc:
        warnings.warn(f"Falling back to default risk-free rate {DEFAULT_RISK_FREE_RATE}: {exc}")
        return DEFAULT_RISK_FREE_RATE


def _clean_quotes(raw: pd.DataFrame, expiry: date, T: float, option_type: str) -> pd.DataFrame:
    """Select/rename columns and derive a market price per quote.

    Yahoo's options feed frequently returns bid=ask=0 (no live market-maker quote) even for
    actively traded contracts. When a live bid/ask exists we use its midpoint (quote_source
    'bid_ask'); otherwise we fall back to the last traded price, but only if the contract
    actually traded today (volume > 0), to avoid pricing off a stale last_price. Rows with
    neither are dropped rather than fabricated.
    """
    quotes = raw[["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].copy()
    quotes = quotes.rename(columns={"openInterest": "open_interest", "lastPrice": "last_price"})
    quotes["expiry"] = expiry
    quotes["T"] = T
    quotes["option_type"] = option_type

    has_live_quote = (quotes["bid"] > 0) & (quotes["ask"] > 0)
    traded_today = quotes["volume"].fillna(0) > 0
    quotes["mid"] = np.where(
        has_live_quote, (quotes["bid"] + quotes["ask"]) / 2.0,
        np.where(traded_today, quotes["last_price"], np.nan),
    )
    quotes["quote_source"] = np.where(has_live_quote, "bid_ask", "last_price")
    return quotes[quotes["mid"] > 0].reset_index(drop=True)


def _cache_path(ticker: str, as_of: date) -> Path:
    """Local cache file for a ticker's chain pull on a given date."""
    return CACHE_DIR / f"{ticker}_{as_of.isoformat()}.csv"


def _select_expiries(available: list[str], as_of: date, max_expiries: int, max_horizon_days: int = 365) -> list[str]:
    """Pick up to max_expiries future expiries evenly spread out to max_horizon_days.

    Tickers with daily-listed expiries (e.g. SPY) would otherwise fill max_expiries with
    consecutive near-dated contracts spanning only a week or two, giving no real maturity
    axis for the surface.
    """
    candidates = [e for e in available if 0 < (date.fromisoformat(e) - as_of).days <= max_horizon_days]
    if not candidates:
        candidates = [e for e in available if (date.fromisoformat(e) - as_of).days > 0]
    if len(candidates) <= max_expiries:
        return candidates
    idx = sorted(set(np.linspace(0, len(candidates) - 1, max_expiries).round().astype(int)))
    return [candidates[i] for i in idx]


def _fetch_chain_live(ticker: str, max_expiries: int, as_of: date, dividend_yield: float) -> pd.DataFrame:
    """Pull spot + raw call/put quotes from yfinance for a maturity-spread set of expiries, cleaned and tagged with T."""
    yft = yf.Ticker(ticker)
    spot = float(yft.history(period="1d")["Close"].iloc[-1])
    rate = get_risk_free_rate()

    expiries = _select_expiries(yft.options, as_of, max_expiries)
    if not expiries:
        raise RuntimeError(f"yfinance returned no listed expirations for {ticker}")

    frames = []
    for expiry_str in expiries:
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        T = (expiry - as_of).days / 365.0
        if T <= 0:
            continue
        opt = yft.option_chain(expiry_str)
        frames.append(_clean_quotes(opt.calls, expiry, T, "call"))
        frames.append(_clean_quotes(opt.puts, expiry, T, "put"))

    if not frames:
        raise RuntimeError(f"No usable (future-dated) expirations returned for {ticker}")

    chain = pd.concat(frames, ignore_index=True)
    chain["spot"], chain["risk_free_rate"], chain["dividend_yield"] = spot, rate, dividend_yield
    return chain


def fetch_chain(
    ticker: str = DEFAULT_TICKER,
    max_expiries: int = 6,
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
    use_cache: bool = True,
) -> MarketSnapshot:
    """Return a MarketSnapshot for `ticker`, pulling live data unless today's pull is already cached."""
    as_of = date.today()
    cache_path = _cache_path(ticker, as_of)

    if use_cache and cache_path.exists():
        chain = pd.read_csv(cache_path, parse_dates=["expiry"])
        chain["expiry"] = chain["expiry"].dt.date
    else:
        chain = _fetch_chain_live(ticker, max_expiries, as_of, dividend_yield)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        chain.to_csv(cache_path, index=False)

    if chain.empty:
        raise RuntimeError(f"No option chain data available for {ticker} on {as_of}")

    return MarketSnapshot(
        ticker=ticker,
        as_of=as_of,
        spot=float(chain["spot"].iloc[0]),
        risk_free_rate=float(chain["risk_free_rate"].iloc[0]),
        dividend_yield=float(chain["dividend_yield"].iloc[0]),
        chain=chain,
    )
