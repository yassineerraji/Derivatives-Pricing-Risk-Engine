"""No-static-arbitrage checks for a calibrated SVI surface: butterfly (per slice) and calendar (across slices)."""

from dataclasses import dataclass, field

import numpy as np

from dpre.calibration.svi import SVIParams, durrleman_g, svi_total_variance


@dataclass
class ButterflyCheck:
    """Result of Durrleman's condition test for one SVI slice: g(k) >= 0 everywhere means no butterfly arbitrage."""

    T: float
    arbitrage_free: bool
    min_g: float


@dataclass
class CalendarViolation:
    """A strike range where total variance decreases from an earlier to a later maturity."""

    T_prev: float
    T_curr: float
    n_violations: int
    worst_gap: float


@dataclass
class CalendarCheck:
    """Result of the calendar-spread check across all consecutive calibrated maturities."""

    arbitrage_free: bool
    violations: list[CalendarViolation] = field(default_factory=list)


def check_butterfly_arbitrage(
    params: SVIParams, k_range: tuple[float, float] = (-1.0, 1.0), n_points: int = 200, tol: float = 1e-6,
) -> ButterflyCheck:
    """Durrleman's condition: the Breeden-Litzenberger implied density stays non-negative iff
    durrleman_g(k, ...) >= 0 for all k (see calibration/svi.py for the formula -- shared with the
    arbitrage-constrained calibration so the checker and the fitter agree on exactly one definition).
    """
    k = np.linspace(*k_range, n_points)
    g = durrleman_g(k, params.a, params.b, params.rho, params.m, params.sigma)
    min_g = float(np.min(g))
    return ButterflyCheck(T=params.T, arbitrage_free=min_g >= -tol, min_g=min_g)


def check_calendar_arbitrage(
    slices: list[SVIParams], spot: float, r: float, q: float, n_strikes: int = 100, tol: float = 1e-8,
) -> CalendarCheck:
    """Total variance at a fixed strike must be non-decreasing in maturity (Roger Lee's no-calendar-arbitrage
    condition). Checked strike-by-strike (not log-moneyness-by-log-moneyness) since each slice has its own
    forward, over the strike range common to each consecutive pair of maturities.
    """
    ordered = sorted(slices, key=lambda p: p.T)
    violations = []
    for prev, curr in zip(ordered, ordered[1:]):
        forward_prev = spot * np.exp((r - q) * prev.T)
        forward_curr = spot * np.exp((r - q) * curr.T)
        lo = max(forward_prev * np.exp(-0.5), forward_curr * np.exp(-0.5))
        hi = min(forward_prev * np.exp(0.5), forward_curr * np.exp(0.5))
        if lo >= hi:
            continue

        strikes = np.linspace(lo, hi, n_strikes)
        w_prev = svi_total_variance(np.log(strikes / forward_prev), prev.a, prev.b, prev.rho, prev.m, prev.sigma)
        w_curr = svi_total_variance(np.log(strikes / forward_curr), curr.a, curr.b, curr.rho, curr.m, curr.sigma)

        bad = w_curr < w_prev - tol
        if bad.any():
            violations.append(
                CalendarViolation(
                    T_prev=prev.T, T_curr=curr.T, n_violations=int(bad.sum()),
                    worst_gap=float((w_prev - w_curr)[bad].max()),
                )
            )
    return CalendarCheck(arbitrage_free=len(violations) == 0, violations=violations)
