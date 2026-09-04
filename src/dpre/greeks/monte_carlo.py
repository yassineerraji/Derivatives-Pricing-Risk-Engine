"""Monte Carlo Greeks: pathwise and likelihood-ratio estimators for delta and vega.

Reuses pricing.monte_carlo.terminal_price/payoff rather than re-simulating GBM paths from
scratch. Pathwise differentiates the payoff (needs it to be a.e. differentiable in S_T, true
for vanilla calls/puts); likelihood-ratio differentiates the terminal-price density instead, so
it works even for non-smooth payoffs, at the cost of a noisier estimator (dividing by sigma or
sigma*sqrt(T) amplifies variance).
"""

from dataclasses import dataclass

import numpy as np

from dpre.pricing.monte_carlo import payoff, terminal_price


@dataclass
class MCGreekResult:
    """A Monte Carlo Greek estimate with its standard error and path count."""

    value: float
    std_error: float
    n_paths: int


def _simulate(S: float, T: float, r: float, sigma: float, q: float, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw standard normals z and the terminal prices they imply."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return terminal_price(S, T, r, sigma, q, z), z


def _dpayoff_dS_T(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """d(payoff)/dS_T for a European vanilla: 1{S_T>K} for a call, -1{S_T<K} for a put."""
    if option_type == "call":
        return (S_T > K).astype(float)
    if option_type == "put":
        return -(S_T < K).astype(float)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _result(discounted: np.ndarray) -> MCGreekResult:
    n_paths = len(discounted)
    return MCGreekResult(value=discounted.mean(), std_error=discounted.std(ddof=1) / np.sqrt(n_paths), n_paths=n_paths)


def pathwise_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, n_paths: int = 100_000, seed: int = 42) -> MCGreekResult:
    """Pathwise delta: E[exp(-rT) * d(payoff)/dS_T * dS_T/dS], with dS_T/dS = S_T/S."""
    S_T, _ = _simulate(S, T, r, sigma, q, n_paths, seed)
    sample = _dpayoff_dS_T(S_T, K, option_type) * (S_T / S)
    return _result(np.exp(-r * T) * sample)


def pathwise_vega(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, n_paths: int = 100_000, seed: int = 42) -> MCGreekResult:
    """Pathwise vega: E[exp(-rT) * d(payoff)/dS_T * dS_T/dsigma], with dS_T/dsigma = S_T*(sqrt(T)*z - sigma*T)."""
    S_T, z = _simulate(S, T, r, sigma, q, n_paths, seed)
    dS_T_dsigma = S_T * (np.sqrt(T) * z - sigma * T)
    sample = _dpayoff_dS_T(S_T, K, option_type) * dS_T_dsigma
    return _result(np.exp(-r * T) * sample)


def likelihood_ratio_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, n_paths: int = 100_000, seed: int = 42) -> MCGreekResult:
    """Likelihood-ratio delta: E[exp(-rT) * payoff(S_T) * score], score = z / (S*sigma*sqrt(T))."""
    S_T, z = _simulate(S, T, r, sigma, q, n_paths, seed)
    score = z / (S * sigma * np.sqrt(T))
    sample = payoff(S_T, K, option_type) * score
    return _result(np.exp(-r * T) * sample)


def likelihood_ratio_vega(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, n_paths: int = 100_000, seed: int = 42) -> MCGreekResult:
    """Likelihood-ratio vega: E[exp(-rT) * payoff(S_T) * score], score = (z^2-1)/sigma - z*sqrt(T)."""
    S_T, z = _simulate(S, T, r, sigma, q, n_paths, seed)
    score = (z**2 - 1) / sigma - z * np.sqrt(T)
    sample = payoff(S_T, K, option_type) * score
    return _result(np.exp(-r * T) * sample)
