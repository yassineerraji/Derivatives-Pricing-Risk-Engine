"""Tests for the delta-hedging simulation: setup, cost accounting, and the frictionless-vs-frictional identity."""

import numpy as np
import pytest

from dpre.calibration.svi import SVIParams
from dpre.risk.book import build_book
from dpre.risk.hedging import HedgeCostModel, run_hedge_simulation, simulate_price_path
from dpre.risk.valuation import book_delta

SPOT, R, Q, SIGMA = 100.0, 0.03, 0.01, 0.2
SVI_SLICES = [
    SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2, T=90 / 365),
    SVIParams(a=0.03, b=0.18, rho=-0.35, m=0.0, sigma=0.22, T=180 / 365),
    SVIParams(a=0.04, b=0.20, rho=-0.3, m=0.0, sigma=0.24, T=270 / 365),
]
N_DAYS = 30
DT = 1 / 365


def test_simulate_price_path_shape_and_start() -> None:
    path = simulate_price_path(SPOT, R, Q, SIGMA, N_DAYS, DT, seed=1)
    assert len(path) == N_DAYS + 1
    assert path[0] == pytest.approx(SPOT)
    assert np.all(path > 0)


def test_initial_hedge_matches_negative_book_delta() -> None:
    book = build_book(SPOT)
    path = simulate_price_path(SPOT, R, Q, SIGMA, N_DAYS, DT, seed=1)
    result = run_hedge_simulation(book, path, R, Q, SVI_SLICES, DT)
    expected = -book_delta(book, SPOT, R, Q, SVI_SLICES, elapsed_years=0.0)
    assert result["hedge_shares"][0] == pytest.approx(expected)


def test_frictionless_default_cost_model_pays_nothing() -> None:
    book = build_book(SPOT)
    path = simulate_price_path(SPOT, R, Q, SIGMA, N_DAYS, DT, seed=1)
    result = run_hedge_simulation(book, path, R, Q, SVI_SLICES, DT, cost_model=None)
    assert np.all(result["cost_paid"] == 0.0)
    assert np.all(result["cumulative_cost"] == 0.0)


def test_frictional_cash_lags_frictionless_by_compounded_cost() -> None:
    """Both runs trade identical share quantities (the cost model doesn't affect the delta target) and
    an identical option/hedge-share path, so cash is the only thing that can differ between them. Since
    cash compounds at the risk-free rate each step, that difference is costs paid so far, individually
    compounded forward from the day each was paid -- not simply their (uncompounded) running sum."""
    book = build_book(SPOT)
    path = simulate_price_path(SPOT, R, Q, SIGMA, N_DAYS, DT, seed=1)
    cost_model = HedgeCostModel(half_spread=0.0005, impact_coefficient=1e-6)

    frictionless = run_hedge_simulation(book, path, R, Q, SVI_SLICES, DT, cost_model=None)
    frictional = run_hedge_simulation(book, path, R, Q, SVI_SLICES, DT, cost_model=cost_model)

    cash_gap = frictionless["cash"] - frictional["cash"]
    t = np.arange(len(path))
    discounted_cost = frictional["cost_paid"] * np.exp(-R * DT * t)
    compounded_cost = np.exp(R * DT * t) * np.cumsum(discounted_cost)

    assert cash_gap == pytest.approx(compounded_cost, abs=1e-6)
    assert np.all(frictional["cumulative_cost"] >= 0)
    assert frictional["cumulative_cost"][-1] > 0  # a nontrivial book should actually trade and pay some cost
    # option_value and hedge_shares should be untouched by the cost model (only cash/cost_paid differ)
    assert frictionless["option_value"] == pytest.approx(frictional["option_value"])
    assert frictionless["hedge_shares"] == pytest.approx(frictional["hedge_shares"])
    # frictional replication should end up strictly worse than frictionless once real costs are paid
    assert frictional["total_pnl"][-1] < frictionless["total_pnl"][-1]


def test_identical_hedge_shares_regardless_of_cost_model() -> None:
    """The traded share quantities shouldn't depend on the cost model, only cash/cost_paid should."""
    book = build_book(SPOT)
    path = simulate_price_path(SPOT, R, Q, SIGMA, N_DAYS, DT, seed=1)
    frictionless = run_hedge_simulation(book, path, R, Q, SVI_SLICES, DT, cost_model=None)
    frictional = run_hedge_simulation(book, path, R, Q, SVI_SLICES, DT, cost_model=HedgeCostModel(0.0005, 1e-6))
    assert frictionless["hedge_shares"] == pytest.approx(frictional["hedge_shares"])
