"""Greeks via bump-and-reprice finite differences on the closed-form Black-Scholes price.

Step sizes below are fixed and documented rather than adaptive, per the bias/variance tradeoff
inherent to bump-and-reprice: a central difference has O(h^2) truncation (bias) error, so
shrinking h reduces bias, but floating-point cancellation in (f(x+h) - f(x-h)) grows as h shrinks,
adding O(machine_eps/h) rounding error. The values below sit near the standard sqrt/cbrt(eps)
sweet spots for first/second derivatives at option-price magnitudes (h scaled by S for delta/gamma
since price is far more sensitive to relative than absolute moves in the underlying).
"""

from dpre.pricing.black_scholes import price as bs_price

H_REL_S = 1e-4
H_SIGMA = 1e-4
H_T = 1e-5
H_R = 1e-5


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, h: float = H_REL_S) -> float:
    """Central-difference delta: (price(S+hS) - price(S-hS)) / (2hS)."""
    bump = h * S
    return (bs_price(S + bump, K, T, r, sigma, option_type, q) - bs_price(S - bump, K, T, r, sigma, option_type, q)) / (2 * bump)


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, h: float = H_REL_S) -> float:
    """Central second-difference gamma: (price(S+hS) - 2*price(S) + price(S-hS)) / (hS)^2.

    Identical for calls and puts (like analytical.gamma), so option_type is fixed internally
    rather than exposed as a no-op parameter.
    """
    bump = h * S
    up = bs_price(S + bump, K, T, r, sigma, "call", q)
    mid = bs_price(S, K, T, r, sigma, "call", q)
    down = bs_price(S - bump, K, T, r, sigma, "call", q)
    return (up - 2 * mid + down) / bump**2


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, h: float = H_SIGMA) -> float:
    """Central-difference vega: (price(sigma+h) - price(sigma-h)) / (2h).

    Identical for calls and puts (like analytical.vega), so option_type is fixed internally
    rather than exposed as a no-op parameter.
    """
    return (bs_price(S, K, T, r, sigma + h, "call", q) - bs_price(S, K, T, r, sigma - h, "call", q)) / (2 * h)


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, h: float = H_T) -> float:
    """Central-difference theta = -d(price)/dT (calendar decay). Assumes T > h."""
    dprice_dT = (bs_price(S, K, T + h, r, sigma, option_type, q) - bs_price(S, K, T - h, r, sigma, option_type, q)) / (2 * h)
    return -dprice_dT


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, h: float = H_R) -> float:
    """Central-difference rho: (price(r+h) - price(r-h)) / (2h)."""
    return (bs_price(S, K, T, r + h, sigma, option_type, q) - bs_price(S, K, T, r - h, sigma, option_type, q)) / (2 * h)
