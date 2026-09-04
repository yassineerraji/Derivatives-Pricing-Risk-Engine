"""Raw SVI (Stochastic Volatility Inspired) smile parameterization and arbitrage-constrained calibration.

calibrate_svi_surface fits slices in increasing maturity order, each one constrained (via SLSQP) to
satisfy Durrleman's no-butterfly-arbitrage condition on its own, and -- given the previously fitted
slice -- to sit at or above it in total variance everywhere (no-calendar-arbitrage), rather than
fitting each slice by plain unconstrained least squares as a first pass did. See
docs/technical_notes.md sec. 3 for why an unconstrained per-slice fit doesn't guarantee this on its
own, and calibration/arbitrage.py for the (now expected-to-pass) post-hoc checks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, least_squares, minimize

DEFAULT_MAX_ABS_LOG_MONEYNESS = 0.5
DEFAULT_ARBITRAGE_CHECK_GRID = np.linspace(-1.5, 1.5, 241)
# Constraints are enforced with a small positive margin, not exactly >= 0: SLSQP itself only
# satisfies constraints to its own solver tolerance, and the constraint grid above is finite while
# calibration/arbitrage.py's post-hoc checkers evaluate on a different (denser, differently ranged)
# grid -- without a margin, a fit that is "just barely" feasible on this grid can land "just barely"
# infeasible (by O(1e-6)) on the checker's. The margin trades a hair of fit quality for headroom.
ARBITRAGE_MARGIN = 1e-3


@dataclass
class SVIParams:
    """Calibrated raw-SVI parameters for one maturity slice, with the total-variance formula in the docstring below."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Raw SVI total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2)), k = log-moneyness ln(K/F)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_implied_vol(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Implied vol at log-moneyness k implied by a calibrated SVI slice: sigma_impl = sqrt(w(k) / T)."""
    w = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
    return np.sqrt(np.maximum(w, 0.0) / params.T)


def durrleman_g(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float, h: float = 1e-4) -> np.ndarray:
    """Durrleman's density-positivity function for a raw SVI slice, evaluated at each k in the array k.

    g(k) >= 0 everywhere is necessary and sufficient for the slice to be free of butterfly arbitrage
    (a negative risk-neutral density somewhere, via Breeden-Litzenberger). w'/w'' are central finite
    differences since svi_total_variance is smooth and cheap to evaluate. Shared by the post-hoc
    checker (calibration/arbitrage.py) and the constrained calibration below, so both use exactly the
    same formula rather than two hand-kept copies of it.
    """
    def w_of(k_: np.ndarray) -> np.ndarray:
        return svi_total_variance(k_, a, b, rho, m, sigma)

    w, w_plus, w_minus = w_of(k), w_of(k + h), w_of(k - h)
    w_prime = (w_plus - w_minus) / (2 * h)
    w_double_prime = (w_plus - 2 * w + w_minus) / h**2
    safe_w = np.where(w > 0, w, 1.0)
    g = (1 - (k * w_prime) / (2 * safe_w)) ** 2 - (w_prime**2 / 4) * (1 / safe_w + 0.25) + w_double_prime / 2
    return np.where(w > 0, g, -1.0)


def calibrate_svi_slice(k: np.ndarray, w: np.ndarray, weights: np.ndarray, T: float) -> SVIParams:
    """Plain (unconstrained) weighted-least-squares SVI fit for one maturity slice. Fast, but -- unlike
    calibrate_svi_surface -- gives no guarantee of butterfly- or calendar-arbitrage-free output; kept
    as the simple building block and for tests that probe fitting behavior in isolation."""
    def residuals(params: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = params
        return np.sqrt(weights) * (svi_total_variance(k, a, b, rho, m, sigma) - w)

    x0 = np.array([max(w.min(), 1e-6), 0.1, 0.0, float(np.median(k)), 0.1])
    bounds = ([-np.inf, 0.0, -0.999, -np.inf, 1e-6], [np.inf, np.inf, 0.999, np.inf, np.inf])
    result = least_squares(residuals, x0, bounds=bounds)
    a, b, rho, m, sigma = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T)


def calibrate_svi_slice_arbitrage_free(
    k: np.ndarray, w: np.ndarray, weights: np.ndarray, T: float,
    prev_slice: SVIParams | None = None, check_grid: np.ndarray = DEFAULT_ARBITRAGE_CHECK_GRID,
) -> SVIParams:
    """Weighted-least-squares SVI fit for one slice, constrained (via SLSQP) so the result satisfies:

    - Durrleman's condition on check_grid (no butterfly arbitrage in this slice on its own).
    - When prev_slice is given, total variance on check_grid is nowhere below prev_slice's (no
      calendar arbitrage against the previous, earlier maturity).

    This generally fits the noisy market quotes slightly worse than the unconstrained
    calibrate_svi_slice -- that's the point: it's trading fit quality for a surface that is actually
    usable for risk without the caveats in docs/technical_notes.md sec. 3.
    """
    def objective(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        residuals = np.sqrt(weights) * (svi_total_variance(k, a, b, rho, m, sigma) - w)
        return float(np.sum(residuals**2))

    def butterfly(params: np.ndarray) -> np.ndarray:
        return durrleman_g(check_grid, *params)

    constraints = [NonlinearConstraint(butterfly, ARBITRAGE_MARGIN, np.inf)]

    if prev_slice is not None:
        prev_w = svi_total_variance(check_grid, prev_slice.a, prev_slice.b, prev_slice.rho, prev_slice.m, prev_slice.sigma)

        def calendar(params: np.ndarray) -> np.ndarray:
            a, b, rho, m, sigma = params
            return svi_total_variance(check_grid, a, b, rho, m, sigma) - prev_w

        constraints.append(NonlinearConstraint(calendar, ARBITRAGE_MARGIN, np.inf))

    x0 = np.array([max(w.min(), 1e-6), 0.1, 0.0, float(np.median(k)), 0.1])
    if prev_slice is not None:
        # Start from a point already satisfying the calendar constraint (prev slice's own shape,
        # nudged up slightly in level) rather than an independent least-squares guess that may not.
        x0 = np.array([prev_slice.a + 1e-4, prev_slice.b, prev_slice.rho, prev_slice.m, prev_slice.sigma])

    bounds = [(-np.inf, np.inf), (1e-8, np.inf), (-0.999, 0.999), (-np.inf, np.inf), (1e-6, np.inf)]
    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    a, b, rho, m, sigma = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma, T=T)


def _slice_weights(group: pd.DataFrame) -> np.ndarray:
    """Weight by inverse bid-ask spread where a live quote exists; last_price-only rows get a wider fallback
    spread (3x the slice's median live spread, or a fixed default if none), since their true spread is unknown."""
    spread = (group["ask"] - group["bid"]).to_numpy()
    is_live = (group.get("quote_source", pd.Series("bid_ask", index=group.index)) == "bid_ask").to_numpy()
    fallback = 3.0 * np.median(spread[is_live]) if is_live.any() else 0.05
    effective_spread = np.where(is_live, spread, fallback)
    return 1.0 / np.maximum(effective_spread, 1e-4)


def _slice_inputs(T: float, group: pd.DataFrame, spot: float, r: float, q: float, max_abs_log_moneyness: float):
    """OTM/ATM-only, moneyness-filtered (k, total_variance, weights) for one maturity slice; None if too few points."""
    forward = spot * np.exp((r - q) * T)
    k_full = np.log(group["strike"].to_numpy() / forward)
    is_otm = np.where(group["option_type"].to_numpy() == "call", k_full >= 0, k_full <= 0)
    in_range = is_otm & (np.abs(k_full) <= max_abs_log_moneyness)
    group, k = group[in_range], k_full[in_range]
    if len(group) < 5:
        return None
    total_var = group["implied_vol"].to_numpy() ** 2 * T
    weights = _slice_weights(group)
    return k, total_var, weights


def calibrate_svi_surface(
    iv_df: pd.DataFrame, spot: float, r: float, q: float,
    max_abs_log_moneyness: float = DEFAULT_MAX_ABS_LOG_MONEYNESS,
) -> list[SVIParams]:
    """Fit one arbitrage-constrained SVI slice per maturity in iv_df (needs strike, T, implied_vol, bid,
    ask columns), in increasing T order so each slice can be constrained against the previous one.

    Two data-quality filters are applied per slice before fitting:
    - OTM/ATM only (calls with strike >= forward, puts with strike <= forward): the ITM leg
      is redundant with the OTM leg under put-call parity but is typically thinner and more
      likely to be priced off a stale last_price, so mixing both legs feeds the fit two
      inconsistent IVs at the same moneyness.
    - |log(K/F)| <= max_abs_log_moneyness: deep-wing strikes are the thinnest quotes of all,
      and raw SVI's five parameters have enough freedom to chase a handful of such outliers
      into a degenerate fit that no longer tracks the liquid part of the smile.
    """
    slices: list[SVIParams] = []
    for T, group in sorted(iv_df.groupby("T"), key=lambda item: item[0]):
        inputs = _slice_inputs(T, group, spot, r, q, max_abs_log_moneyness)
        if inputs is None:
            continue
        k, total_var, weights = inputs
        prev_slice = slices[-1] if slices else None
        slices.append(calibrate_svi_slice_arbitrage_free(k, total_var, weights, T, prev_slice=prev_slice))
    return slices
