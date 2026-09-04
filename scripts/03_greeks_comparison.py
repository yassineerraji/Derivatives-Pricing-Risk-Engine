"""Deliverable: accuracy and computation-cost comparison of analytical, finite-difference, and Monte Carlo Greeks."""

import time
from pathlib import Path

import pandas as pd

from dpre.greeks import analytical, finite_difference
from dpre.greeks.monte_carlo import likelihood_ratio_delta, likelihood_ratio_vega, pathwise_delta, pathwise_vega

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR = RESULTS_DIR / "tables"

SPOT, RATE, DIV, SIGMA, MATURITY = 100.0, 0.03, 0.01, 0.20, 1.0
STRIKE = 100.0
OPTION_TYPES = ["call", "put"]
N_PATHS_MC = 200_000
SEED = 42


def _timed(fn, repeats: int = 5):
    """Run fn() `repeats` times, returning (best elapsed seconds, result of the last call)."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def build_greeks_comparison_table() -> pd.DataFrame:
    """One row per (option_type, greek, method): value, error vs analytical, MC std error, and runtime."""
    rows = []
    for option_type in OPTION_TYPES:
        analytical_fns = {
            "delta": lambda: analytical.delta(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV),
            "gamma": lambda: analytical.gamma(SPOT, STRIKE, MATURITY, RATE, SIGMA, DIV),
            "vega": lambda: analytical.vega(SPOT, STRIKE, MATURITY, RATE, SIGMA, DIV),
            "theta": lambda: analytical.theta(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV),
            "rho": lambda: analytical.rho(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV),
        }
        fd_fns = {
            "delta": lambda: finite_difference.delta(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV),
            "gamma": lambda: finite_difference.gamma(SPOT, STRIKE, MATURITY, RATE, SIGMA, DIV),
            "vega": lambda: finite_difference.vega(SPOT, STRIKE, MATURITY, RATE, SIGMA, DIV),
            "theta": lambda: finite_difference.theta(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV),
            "rho": lambda: finite_difference.rho(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV),
        }
        mc_fns = {
            ("delta", "mc_pathwise"): lambda: pathwise_delta(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV, n_paths=N_PATHS_MC, seed=SEED),
            ("delta", "mc_likelihood_ratio"): lambda: likelihood_ratio_delta(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV, n_paths=N_PATHS_MC, seed=SEED),
            ("vega", "mc_pathwise"): lambda: pathwise_vega(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV, n_paths=N_PATHS_MC, seed=SEED),
            ("vega", "mc_likelihood_ratio"): lambda: likelihood_ratio_vega(SPOT, STRIKE, MATURITY, RATE, SIGMA, option_type, DIV, n_paths=N_PATHS_MC, seed=SEED),
        }

        analytical_values = {}
        for greek, fn in analytical_fns.items():
            t, value = _timed(fn)
            analytical_values[greek] = value
            rows.append({"option_type": option_type, "greek": greek, "method": "analytical", "value": value, "error": 0.0, "std_error": None, "time_s": t})

        for greek, fn in fd_fns.items():
            t, value = _timed(fn)
            rows.append({"option_type": option_type, "greek": greek, "method": "finite_difference", "value": value, "error": abs(value - analytical_values[greek]), "std_error": None, "time_s": t})

        for (greek, method), fn in mc_fns.items():
            t, result = _timed(fn)
            rows.append({"option_type": option_type, "greek": greek, "method": method, "value": result.value, "error": abs(result.value - analytical_values[greek]), "std_error": result.std_error, "time_s": t})

    return pd.DataFrame(rows)


def main() -> None:
    """Generate the Phase 3 Greeks deliverable: accuracy/cost comparison across analytical, FD, and MC methods."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table = build_greeks_comparison_table()
    table.to_csv(TABLES_DIR / "greeks_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"Saved deliverable to {TABLES_DIR}")


if __name__ == "__main__":
    main()
