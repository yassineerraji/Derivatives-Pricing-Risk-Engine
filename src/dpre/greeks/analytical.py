"""Closed-form Black-Scholes Greeks for European calls and puts."""

import numpy as np
from scipy.stats import norm

from dpre.pricing.black_scholes import d1_d2


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    """d(price)/dS. Call: exp(-qT)*N(d1); put: exp(-qT)*(N(d1)-1)."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    if option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """d^2(price)/dS^2, identical for calls and puts."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """d(price)/d(sigma), identical for calls and puts. Units: price change per 1.0 (100%) change in vol."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    """d(price)/dt (calendar theta, decay per year of passing time; typically negative for long options)."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return decay - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    if option_type == "put":
        return decay + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    """d(price)/dr. Units: price change per 1.0 (100%) change in the risk-free rate."""
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
