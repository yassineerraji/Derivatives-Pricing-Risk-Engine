"""Accuracy tests for bump-and-reprice Greeks against the closed-form analytical values."""

import pytest

from dpre.greeks import analytical, finite_difference

S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.03, 0.22, 0.015
OPTION_TYPES = ["call", "put"]
TOLERANCE = 1e-4


@pytest.mark.parametrize("option_type", OPTION_TYPES)
@pytest.mark.parametrize("greek_name", ["delta", "theta", "rho"])
def test_fd_greek_matches_analytical(option_type: str, greek_name: str) -> None:
    """Each option-type-dependent bump-and-reprice Greek matches its closed-form counterpart within TOLERANCE."""
    expected = getattr(analytical, greek_name)(S, K, T, r, sigma, option_type, q)
    actual = getattr(finite_difference, greek_name)(S, K, T, r, sigma, option_type, q)
    assert actual == pytest.approx(expected, abs=TOLERANCE)


@pytest.mark.parametrize("greek_name", ["gamma", "vega"])
def test_fd_greek_matches_analytical_no_option_type(greek_name: str) -> None:
    """gamma/vega are identical for calls and puts in both modules, so neither takes option_type."""
    expected = getattr(analytical, greek_name)(S, K, T, r, sigma, q)
    actual = getattr(finite_difference, greek_name)(S, K, T, r, sigma, q)
    assert actual == pytest.approx(expected, abs=TOLERANCE)
