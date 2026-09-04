"""Book valuation and delta: prices each position off the calibrated SVI surface rather than a flat vol.

S may be a scalar or a numpy array (one value per scenario/path point) throughout; every function
here is written so that vectorizes automatically through numpy broadcasting.
"""

import numpy as np

from dpre.calibration.svi import SVIParams, svi_implied_vol
from dpre.greeks.analytical import delta as bs_delta
from dpre.pricing.black_scholes import price as bs_price
from dpre.risk.book import Position


def nearest_slice(T: float, svi_slices: list[SVIParams]) -> SVIParams:
    """The calibrated SVI slice whose maturity is closest to T; used as a fixed smile-shape proxy for T."""
    return min(svi_slices, key=lambda p: abs(p.T - T))


def vol_for(position: Position, S, r: float, q: float, svi_slices: list[SVIParams]):
    """Implied vol at spot S for `position`'s strike, from the SVI slice nearest its own maturity.

    This is a "sticky-strike" assumption: the smile SHAPE (the calibrated slice) is frozen at its
    Phase 2 calibration, and only the position's moneyness relative to the (moving) forward decides
    where on that frozen smile its vol is read off. The risk simulation never re-runs SVI calibration.
    """
    slice_ = nearest_slice(position.T, svi_slices)
    forward = S * np.exp((r - q) * slice_.T)
    k = np.log(position.strike / forward)
    return svi_implied_vol(k, slice_)


def position_value(position: Position, S, r: float, q: float, svi_slices: list[SVIParams], T_remaining: float):
    """Mark-to-market value (price * quantity) of one position at spot S with T_remaining years left."""
    vol = vol_for(position, S, r, q, svi_slices)
    return position.quantity * bs_price(S, position.strike, T_remaining, r, vol, position.option_type, q)


def position_delta(position: Position, S, r: float, q: float, svi_slices: list[SVIParams], T_remaining: float):
    """Position delta (Greek * quantity) at spot S with T_remaining years left."""
    vol = vol_for(position, S, r, q, svi_slices)
    return position.quantity * bs_delta(S, position.strike, T_remaining, r, vol, position.option_type, q)


def book_value(positions: list[Position], S, r: float, q: float, svi_slices: list[SVIParams], elapsed_years: float = 0.0):
    """Sum of position_value across the book, each position's own T reduced by elapsed_years."""
    return sum(position_value(p, S, r, q, svi_slices, p.T - elapsed_years) for p in positions)


def book_delta(positions: list[Position], S, r: float, q: float, svi_slices: list[SVIParams], elapsed_years: float = 0.0):
    """Sum of position_delta across the book, each position's own T reduced by elapsed_years."""
    return sum(position_delta(p, S, r, q, svi_slices, p.T - elapsed_years) for p in positions)
