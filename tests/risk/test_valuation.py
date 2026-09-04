"""Tests for book/position valuation against a synthetic (but realistic) calibrated SVI surface."""

import numpy as np
import pytest

from dpre.calibration.svi import SVIParams
from dpre.pricing.black_scholes import price as bs_price
from dpre.risk.book import Position, build_book
from dpre.risk.valuation import book_delta, book_representative_vol, book_value, nearest_slice, position_value, vol_for

SPOT, R, Q = 100.0, 0.03, 0.01
SVI_SLICES = [
    SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2, T=90 / 365),
    SVIParams(a=0.03, b=0.18, rho=-0.35, m=0.0, sigma=0.22, T=180 / 365),
    SVIParams(a=0.04, b=0.20, rho=-0.3, m=0.0, sigma=0.24, T=270 / 365),
]


def test_nearest_slice_picks_closest_maturity() -> None:
    assert nearest_slice(90 / 365, SVI_SLICES).T == pytest.approx(90 / 365)
    assert nearest_slice(0.65, SVI_SLICES).T == pytest.approx(270 / 365)
    assert nearest_slice(0.001, SVI_SLICES).T == pytest.approx(90 / 365)


def test_position_value_matches_black_scholes_with_svi_vol() -> None:
    """position_value should equal quantity * BS price using exactly the vol vol_for() reports."""
    position = Position("call", 100.0, 90 / 365, 10.0)
    vol = float(vol_for(position, SPOT, R, Q, SVI_SLICES))
    expected = 10.0 * bs_price(SPOT, 100.0, 90 / 365, R, vol, "call", Q)
    actual = position_value(position, SPOT, R, Q, SVI_SLICES, T_remaining=90 / 365)
    assert actual == pytest.approx(expected)


def test_position_value_vectorizes_over_spot_array() -> None:
    """S can be a numpy array of scenarios; output should match looping scalar-by-scalar."""
    position = Position("put", 95.0, 180 / 365, -5.0)
    spots = np.array([80.0, 95.0, 100.0, 120.0])
    vectorized = position_value(position, spots, R, Q, SVI_SLICES, T_remaining=180 / 365)
    looped = np.array([position_value(position, float(s), R, Q, SVI_SLICES, T_remaining=180 / 365) for s in spots])
    assert vectorized == pytest.approx(looped)


def test_book_value_is_sum_of_position_values() -> None:
    book = build_book(SPOT)
    total = book_value(book, SPOT, R, Q, SVI_SLICES)
    manual_sum = sum(position_value(p, SPOT, R, Q, SVI_SLICES, T_remaining=p.T) for p in book)
    assert total == pytest.approx(manual_sum)


def test_book_delta_reduces_time_remaining_by_elapsed_years() -> None:
    book = build_book(SPOT)
    delta_now = book_delta(book, SPOT, R, Q, SVI_SLICES, elapsed_years=0.0)
    delta_later = book_delta(book, SPOT, R, Q, SVI_SLICES, elapsed_years=0.05)
    assert delta_now != pytest.approx(delta_later)


def test_book_representative_vol_is_within_the_range_of_position_vols() -> None:
    """A vega-weighted average must lie within [min, max] of the individual positions' own vols."""
    book = build_book(SPOT)
    individual_vols = [float(vol_for(p, SPOT, R, Q, SVI_SLICES)) for p in book]
    rep_vol = book_representative_vol(book, SPOT, R, Q, SVI_SLICES)
    assert min(individual_vols) <= rep_vol <= max(individual_vols)


def test_book_representative_vol_matches_single_position_vol() -> None:
    """With one position, the vega-weighted average is trivially just that position's own vol."""
    book = [Position("call", 100.0, 90 / 365, 10.0)]
    expected = float(vol_for(book[0], SPOT, R, Q, SVI_SLICES))
    assert book_representative_vol(book, SPOT, R, Q, SVI_SLICES) == pytest.approx(expected)
