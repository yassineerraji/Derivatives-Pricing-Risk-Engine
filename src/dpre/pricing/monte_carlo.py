"""Monte Carlo pricing for European options under risk-neutral GBM, with antithetic and control variates."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    """A Monte Carlo price estimate: the price, its standard error, a 95% CI, sample variance, and path count."""

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    variance: float
    n_paths: int


def terminal_price(S: float, T: float, r: float, sigma: float, q: float, z: np.ndarray) -> np.ndarray:
    """Map standard normal draws z to terminal stock prices under risk-neutral GBM."""
    drift = (r - q - 0.5 * sigma**2) * T
    return S * np.exp(drift + sigma * np.sqrt(T) * z)


def payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Terminal payoff of a European call or put given simulated terminal prices."""
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    if option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _apply_control_variate(payoff: np.ndarray, S_T: np.ndarray, control_mean: float) -> np.ndarray:
    """Adjust payoff draws using S_T as a control variate with known mean E[S_T] = control_mean."""
    b = np.cov(payoff, S_T, ddof=1)[0, 1] / np.var(S_T, ddof=1)
    return payoff - b * (S_T - control_mean)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
    n_paths: int = 10_000,
    seed: int = 42,
    antithetic: bool = True,
    control_variate: bool = True,
) -> MCResult:
    """Price a European option by Monte Carlo. Toggle antithetic/control_variate independently to compare variance reduction."""
    rng = np.random.default_rng(seed)
    control_mean = S * np.exp((r - q) * T)

    if antithetic:
        half = (n_paths + 1) // 2
        z = rng.standard_normal(half)
        S_T_plus = terminal_price(S, T, r, sigma, q, z)
        S_T_minus = terminal_price(S, T, r, sigma, q, -z)
        payoff_plus = payoff(S_T_plus, K, option_type)
        payoff_minus = payoff(S_T_minus, K, option_type)
        if control_variate:
            payoff_plus = _apply_control_variate(payoff_plus, S_T_plus, control_mean)
            payoff_minus = _apply_control_variate(payoff_minus, S_T_minus, control_mean)
        sample = (payoff_plus + payoff_minus) / 2.0
        total_paths = 2 * half
    else:
        z = rng.standard_normal(n_paths)
        S_T = terminal_price(S, T, r, sigma, q, z)
        sample = payoff(S_T, K, option_type)
        if control_variate:
            sample = _apply_control_variate(sample, S_T, control_mean)
        total_paths = n_paths

    discounted = np.exp(-r * T) * sample
    n_eff = len(discounted)
    price = discounted.mean()
    variance = discounted.var(ddof=1)
    std_error = np.sqrt(variance / n_eff)
    return MCResult(
        price=price,
        std_error=std_error,
        ci_low=price - 1.96 * std_error,
        ci_high=price + 1.96 * std_error,
        variance=variance,
        n_paths=total_paths,
    )
