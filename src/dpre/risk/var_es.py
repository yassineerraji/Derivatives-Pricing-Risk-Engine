"""Historical-simulation and Monte Carlo VaR / Expected Shortfall for the option book, via full revaluation."""

from dataclasses import dataclass

import numpy as np
import yfinance as yf

from dpre.calibration.svi import SVIParams, svi_implied_vol
from dpre.risk.book import Position
from dpre.risk.valuation import book_value

DAYS_PER_YEAR = 365  # elapsed trading days approximated as calendar days, consistent with the ACT/365
                      # day-count convention used throughout the project (see CLAUDE.md)


@dataclass
class VaRResult:
    """VaR/ES at one confidence level, in book P&L currency units, with a bootstrap 95% CI on each."""

    confidence: float
    var: float
    es: float
    var_ci: tuple[float, float]
    es_ci: tuple[float, float]
    n_scenarios: int


def fetch_historical_log_returns(ticker: str, lookback_days: int = 250) -> np.ndarray:
    """Daily log returns of `ticker`'s close over roughly the last lookback_days trading days."""
    history = yf.Ticker(ticker).history(period=f"{lookback_days + 30}d")["Close"]
    if len(history) < lookback_days + 1:
        raise RuntimeError(f"Not enough price history for {ticker}: got {len(history)} days, need {lookback_days + 1}")
    log_returns = np.diff(np.log(history.to_numpy()))
    return log_returns[-lookback_days:]


def _var_es_from_pnl(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    """VaR = -quantile of P&L at (1-confidence); ES = -mean of P&L in the tail at or beyond that quantile."""
    threshold = np.quantile(pnl, 1 - confidence)
    tail = pnl[pnl <= threshold]
    return -float(threshold), -float(tail.mean())


def _bootstrap_ci(pnl: np.ndarray, confidence: float, n_boot: int = 1000, seed: int = 42) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap 95% CI for VaR and ES by resampling the P&L scenarios (with replacement) n_boot times."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    var_samples = np.empty(n_boot)
    es_samples = np.empty(n_boot)
    for i in range(n_boot):
        resample = pnl[rng.integers(0, n, n)]
        var_samples[i], es_samples[i] = _var_es_from_pnl(resample, confidence)
    var_ci = (float(np.percentile(var_samples, 2.5)), float(np.percentile(var_samples, 97.5)))
    es_ci = (float(np.percentile(es_samples, 2.5)), float(np.percentile(es_samples, 97.5)))
    return var_ci, es_ci


def _revalue_pnl(
    positions: list[Position], base_value: float, S0: float, log_returns: np.ndarray,
    r: float, q: float, svi_slices: list[SVIParams], horizon_days: int,
) -> np.ndarray:
    """Book P&L (shocked - base) for each return scenario, with every position's T reduced by horizon_days."""
    shocked_spots = S0 * np.exp(log_returns)
    elapsed_years = horizon_days / DAYS_PER_YEAR
    shocked_values = book_value(positions, shocked_spots, r, q, svi_slices, elapsed_years=elapsed_years)
    return shocked_values - base_value


def historical_var_es(
    positions: list[Position], S0: float, r: float, q: float, svi_slices: list[SVIParams],
    ticker: str = "SPY", lookback_days: int = 250, confidence: float = 0.95, horizon_days: int = 1,
) -> VaRResult:
    """Full-revaluation historical VaR/ES: replay each historical daily return against today's book.

    Single-day historical log returns are scaled to the horizon by sqrt(horizon_days) (the standard
    i.i.d.-returns approximation), rather than resampling overlapping multi-day historical windows.
    """
    log_returns = fetch_historical_log_returns(ticker, lookback_days) * np.sqrt(horizon_days)
    base_value = book_value(positions, S0, r, q, svi_slices)
    pnl = _revalue_pnl(positions, base_value, S0, log_returns, r, q, svi_slices, horizon_days)
    var, es = _var_es_from_pnl(pnl, confidence)
    var_ci, es_ci = _bootstrap_ci(pnl, confidence)
    return VaRResult(confidence, var, es, var_ci, es_ci, len(pnl))


def monte_carlo_var_es(
    positions: list[Position], S0: float, r: float, q: float, svi_slices: list[SVIParams],
    confidence: float = 0.95, horizon_days: int = 1, n_scenarios: int = 20_000, seed: int = 42,
) -> VaRResult:
    """Full-revaluation MC VaR/ES: simulate horizon-day GBM returns at the calibrated near-term ATM vol."""
    dt = horizon_days / DAYS_PER_YEAR
    atm_slice = min(svi_slices, key=lambda p: abs(p.T - dt))
    sigma = float(svi_implied_vol(0.0, atm_slice))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_scenarios)
    log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    base_value = book_value(positions, S0, r, q, svi_slices)
    pnl = _revalue_pnl(positions, base_value, S0, log_returns, r, q, svi_slices, horizon_days)
    var, es = _var_es_from_pnl(pnl, confidence)
    var_ci, es_ci = _bootstrap_ci(pnl, confidence)
    return VaRResult(confidence, var, es, var_ci, es_ci, len(pnl))
