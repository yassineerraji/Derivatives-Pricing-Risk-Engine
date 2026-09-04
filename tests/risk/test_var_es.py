"""Tests for the VaR/ES math (network-free: monte_carlo_var_es needs no external data)."""

import numpy as np
import pytest

from dpre.calibration.svi import SVIParams
from dpre.risk.book import build_book
from dpre.risk.var_es import _bootstrap_ci, _var_es_from_pnl, monte_carlo_var_es

SPOT, R, Q = 100.0, 0.03, 0.01
SVI_SLICES = [
    SVIParams(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2, T=90 / 365),
    SVIParams(a=0.03, b=0.18, rho=-0.35, m=0.0, sigma=0.22, T=180 / 365),
    SVIParams(a=0.04, b=0.20, rho=-0.3, m=0.0, sigma=0.24, T=270 / 365),
]


def test_var_es_from_pnl_on_known_distribution() -> None:
    """100 evenly spaced P&L outcomes from -99 to 0: the 5% quantile is the 5th worst outcome."""
    pnl = np.arange(-99, 1, 1.0)  # -99, -98, ..., 0 (100 values)
    var, es = _var_es_from_pnl(pnl, confidence=0.95)
    threshold = np.quantile(pnl, 0.05)
    expected_es = -pnl[pnl <= threshold].mean()
    assert var == pytest.approx(-threshold)
    assert es == pytest.approx(expected_es)
    assert es >= var  # ES (tail average) is at least as severe as VaR (tail boundary)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    pnl = rng.normal(loc=0.0, scale=10.0, size=2000)
    var, es = _var_es_from_pnl(pnl, confidence=0.95)
    var_ci, es_ci = _bootstrap_ci(pnl, confidence=0.95, n_boot=200)
    assert var_ci[0] <= var <= var_ci[1]
    assert es_ci[0] <= es <= es_ci[1]


def test_mc_var_es_higher_confidence_gives_larger_risk_measures() -> None:
    """99% VaR/ES should be at least as large as 95% VaR/ES for the same book and scenarios."""
    book = build_book(SPOT)
    result_95 = monte_carlo_var_es(book, SPOT, R, Q, SVI_SLICES, confidence=0.95, n_scenarios=5000)
    result_99 = monte_carlo_var_es(book, SPOT, R, Q, SVI_SLICES, confidence=0.99, n_scenarios=5000)
    assert result_99.var >= result_95.var
    assert result_99.es >= result_95.es


def test_mc_var_es_es_at_least_var() -> None:
    book = build_book(SPOT)
    result = monte_carlo_var_es(book, SPOT, R, Q, SVI_SLICES, confidence=0.95, n_scenarios=5000)
    assert result.es >= result.var
    assert result.var_ci[0] <= result.var <= result.var_ci[1]
    assert result.es_ci[0] <= result.es <= result.es_ci[1]
