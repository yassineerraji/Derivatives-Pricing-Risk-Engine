"""Crank-Nicolson finite-difference solver for the Black-Scholes PDE, European options."""

import numpy as np
from scipy.linalg import solve_banded


def price_crank_nicolson(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
    M: int = 400,
    N: int = 400,
    S_max_mult: float = 4.0,
) -> float:
    """Price a European option on a uniform S-grid via Crank-Nicolson. M=space steps, N=time steps."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    S_max = S_max_mult * max(S, K)
    dS = S_max / M
    dt = T / N
    i = np.arange(0, M + 1)
    S_grid = i * dS

    is_call = option_type == "call"
    V = np.maximum(S_grid - K, 0.0) if is_call else np.maximum(K - S_grid, 0.0)

    # Tridiagonal coefficients (Wilmott discretization) for interior nodes i=1..M-1.
    a = 0.25 * dt * (sigma**2 * i**2 - (r - q) * i)
    b = -0.5 * dt * (sigma**2 * i**2 + r)
    c = 0.25 * dt * (sigma**2 * i**2 + (r - q) * i)
    a_int, b_int, c_int = a[1:M], b[1:M], c[1:M]

    ab = np.zeros((3, M - 1))
    ab[0, 1:] = -c_int[:-1]
    ab[1, :] = 1.0 - b_int
    ab[2, :-1] = -a_int[1:]

    for n in range(N):
        tau_new = (n + 1) * dt
        if is_call:
            V0_new, VM_new = 0.0, S_max * np.exp(-q * tau_new) - K * np.exp(-r * tau_new)
        else:
            V0_new, VM_new = K * np.exp(-r * tau_new), 0.0

        rhs = a_int * V[0:M - 1] + (1.0 + b_int) * V[1:M] + c_int * V[2:M + 1]
        rhs[0] += a_int[0] * V0_new
        rhs[-1] += c_int[-1] * VM_new

        V[1:M] = solve_banded((1, 1), ab, rhs)
        V[0], V[M] = V0_new, VM_new

    return float(np.interp(S, S_grid, V))
