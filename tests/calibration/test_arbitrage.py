"""Tests for the butterfly and calendar no-arbitrage checks against known good/bad SVI slices."""

from dpre.calibration.arbitrage import check_butterfly_arbitrage, check_calendar_arbitrage
from dpre.calibration.svi import SVIParams

WELL_BEHAVED = dict(a=0.02, b=0.15, rho=-0.4, m=0.0, sigma=0.2)


def test_well_behaved_slice_passes_butterfly_check() -> None:
    """A mild, realistic SVI slice should satisfy Durrleman's condition everywhere."""
    params = SVIParams(**WELL_BEHAVED, T=0.5)
    result = check_butterfly_arbitrage(params)
    assert result.arbitrage_free
    assert result.min_g >= 0


def test_degenerate_slice_fails_butterfly_check() -> None:
    """A pathologically steep slice (large b, |rho| near 1) should violate Durrleman's condition."""
    params = SVIParams(a=-0.05, b=100.0, rho=0.999, m=0.25, sigma=0.005, T=0.1)
    result = check_butterfly_arbitrage(params)
    assert not result.arbitrage_free
    assert result.min_g < 0


def test_negative_total_variance_fails_butterfly_check() -> None:
    """A slice whose minimum total variance dips below zero is flagged directly, without needing Durrleman's g(k)."""
    params = SVIParams(a=-1.0, b=0.01, rho=0.0, m=0.0, sigma=0.01, T=0.1)
    result = check_butterfly_arbitrage(params)
    assert not result.arbitrage_free


def test_monotonic_variance_passes_calendar_check() -> None:
    """Two slices with the later maturity uniformly wider than the earlier one should show no calendar arbitrage."""
    early = SVIParams(a=0.02, b=0.15, rho=-0.3, m=0.0, sigma=0.2, T=0.25)
    late = SVIParams(a=0.05, b=0.20, rho=-0.3, m=0.0, sigma=0.2, T=1.0)
    result = check_calendar_arbitrage([early, late], spot=100.0, r=0.03, q=0.01)
    assert result.arbitrage_free
    assert result.violations == []


def test_decreasing_variance_fails_calendar_check() -> None:
    """A later maturity with strictly lower total variance than the earlier one is a textbook calendar violation."""
    early = SVIParams(a=0.08, b=0.15, rho=-0.3, m=0.0, sigma=0.2, T=0.25)
    late = SVIParams(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.2, T=1.0)
    result = check_calendar_arbitrage([early, late], spot=100.0, r=0.03, q=0.01)
    assert not result.arbitrage_free
    assert len(result.violations) == 1
    assert result.violations[0].T_prev == 0.25
    assert result.violations[0].T_curr == 1.0
