"""Tests for the multi-path P&L attribution orchestration."""

import numpy as np
import pytest

from dpre.calibration.svi import SVIParams
from dpre.risk.book import build_book
from dpre.risk.hedging import HedgeCostModel
from dpre.risk.pnl import run_pnl_attribution

SPOT, R, Q = 100.0, 0.03, 0.01
SVI_SLICES = [
    SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2, T=90 / 365),
    SVIParams(a=0.03, b=0.18, rho=-0.35, m=0.0, sigma=0.22, T=180 / 365),
    SVIParams(a=0.04, b=0.20, rho=-0.3, m=0.0, sigma=0.24, T=270 / 365),
]


def test_pnl_attribution_shapes_and_cost_sign() -> None:
    book = build_book(SPOT)
    cost_model = HedgeCostModel(half_spread=0.0005, impact_coefficient=1e-6)
    n_paths, n_days = 20, 30
    result = run_pnl_attribution(book, SPOT, R, Q, SVI_SLICES, cost_model, n_days=n_days, n_paths=n_paths, seed=1)

    expected_shape = (n_paths, n_days + 1)
    assert result.price_paths.shape == expected_shape
    assert result.frictionless_pnl.shape == expected_shape
    assert result.frictional_pnl.shape == expected_shape
    assert result.cumulative_cost.shape == expected_shape
    assert len(result.days) == n_days + 1

    assert result.price_paths[:, 0] == pytest.approx(SPOT)
    assert np.all(np.diff(result.cumulative_cost, axis=1) >= -1e-9)  # non-decreasing along each path
    assert np.all(result.cumulative_cost[:, -1] > 0)
    # on average, frictional replication should end up worse than frictionless once real costs are paid
    assert result.frictional_pnl[:, -1].mean() < result.frictionless_pnl[:, -1].mean()


def test_pnl_attribution_paths_are_independent() -> None:
    """Different paths within one run shouldn't be identical (each draws its own seed)."""
    book = build_book(SPOT)
    cost_model = HedgeCostModel(half_spread=0.0005, impact_coefficient=1e-6)
    result = run_pnl_attribution(book, SPOT, R, Q, SVI_SLICES, cost_model, n_days=20, n_paths=10, seed=1)
    assert not np.allclose(result.price_paths[0], result.price_paths[1])


def test_pnl_attribution_reproducible_with_same_seed() -> None:
    book = build_book(SPOT)
    cost_model = HedgeCostModel(half_spread=0.0005, impact_coefficient=1e-6)
    a = run_pnl_attribution(book, SPOT, R, Q, SVI_SLICES, cost_model, n_days=20, n_paths=10, seed=7)
    b = run_pnl_attribution(book, SPOT, R, Q, SVI_SLICES, cost_model, n_days=20, n_paths=10, seed=7)
    assert np.array_equal(a.price_paths, b.price_paths)
    assert np.array_equal(a.frictional_pnl, b.frictional_pnl)
