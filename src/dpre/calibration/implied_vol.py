"""Implied volatility extraction: Brent's method (default, robust) and Newton-Raphson (documented alternative)."""

import warnings

import pandas as pd
from scipy.optimize import brentq

from dpre.calibration.data import MarketSnapshot
from dpre.greeks.analytical import vega
from dpre.pricing.black_scholes import price as bs_price


def implied_vol_brent(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str,
    q: float = 0.0, lo: float = 1e-6, hi: float = 5.0,
) -> float:
    """Solve for sigma such that bs_price(...) == market_price via bracketed Brent search on [lo, hi]."""
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    if objective(lo) * objective(hi) > 0:
        raise ValueError(f"No bracketing root in sigma in [{lo}, {hi}] for market_price={market_price}")
    return brentq(objective, lo, hi, xtol=1e-8)


def implied_vol_newton(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str,
    q: float = 0.0, initial_guess: float = 0.2, tol: float = 1e-8, max_iter: int = 100,
) -> float:
    """Solve for sigma via Newton-Raphson using analytical vega. Faster than Brent but can diverge far from the root."""
    sigma = initial_guess
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option_type, q) - market_price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma, q)
        if v < 1e-12:
            raise RuntimeError("Vega too small for Newton-Raphson to proceed; use implied_vol_brent instead")
        sigma = max(sigma - diff / v, 1e-6)
    raise RuntimeError(f"Newton-Raphson did not converge within {max_iter} iterations")


def extract_iv_surface(snapshot: MarketSnapshot, method: str = "brent") -> pd.DataFrame:
    """Attach an implied_vol column to snapshot.chain, solved from the mid price. Unsolvable rows are dropped."""
    solver = implied_vol_brent if method == "brent" else implied_vol_newton
    rows, n_failed = [], 0
    for row in snapshot.chain.itertuples(index=False):
        try:
            iv = solver(
                row.mid, snapshot.spot, row.strike, row.T, snapshot.risk_free_rate, row.option_type,
                snapshot.dividend_yield,
            )
        except (ValueError, RuntimeError):
            n_failed += 1
            continue
        rows.append({**row._asdict(), "implied_vol": iv})
    if n_failed:
        warnings.warn(f"Dropped {n_failed}/{len(snapshot.chain)} quotes that failed to solve for implied vol")
    return pd.DataFrame(rows)
