"""Closed-form Black-Scholes-Merton pricing for European options. Validation baseline for all other methods."""

import numpy as np
from scipy.stats import norm


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    """Return (d1, d2) for the given inputs. T in years, r/q continuously compounded, sigma annualized."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Price a European call. S=spot, K=strike, T=maturity in years, r=risk-free rate, sigma=annualized vol, q=dividend yield."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Price a European put. Same conventions as call_price."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def price(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    """Dispatch to call_price or put_price. option_type is 'call' or 'put'."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma, q)
    if option_type == "put":
        return put_price(S, K, T, r, sigma, q)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
