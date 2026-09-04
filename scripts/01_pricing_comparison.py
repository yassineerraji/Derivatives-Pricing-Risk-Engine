"""Deliverable: pricing comparison table (BS/MC/FD) and Monte Carlo convergence/variance-reduction analysis."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dpre.pricing.black_scholes import price as bs_price
from dpre.pricing.finite_difference import price_crank_nicolson
from dpre.pricing.monte_carlo import mc_price

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

SPOT, RATE, DIV, SIGMA, MATURITY = 100.0, 0.03, 0.01, 0.20, 1.0
STRIKES = [80.0, 90.0, 100.0, 110.0, 120.0]
OPTION_TYPES = ["call", "put"]
N_PATHS_TABLE = 200_000
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


def build_pricing_comparison_table() -> pd.DataFrame:
    """Compute BS/MC/FD price, runtime, and error vs BS across strikes and option types."""
    rows = []
    for option_type in OPTION_TYPES:
        for K in STRIKES:
            bs_time, bs = _timed(lambda: bs_price(SPOT, K, MATURITY, RATE, SIGMA, option_type, DIV))
            mc_time, mc = _timed(
                lambda: mc_price(SPOT, K, MATURITY, RATE, SIGMA, option_type, DIV, n_paths=N_PATHS_TABLE, seed=SEED)
            )
            fd_time, fd = _timed(lambda: price_crank_nicolson(SPOT, K, MATURITY, RATE, SIGMA, option_type, DIV))
            rows.append(
                {
                    "option_type": option_type,
                    "strike": K,
                    "bs_price": bs,
                    "mc_price": mc.price,
                    "mc_ci_low": mc.ci_low,
                    "mc_ci_high": mc.ci_high,
                    "fd_price": fd,
                    "mc_error": abs(mc.price - bs),
                    "fd_error": abs(fd - bs),
                    "bs_time_s": bs_time,
                    "mc_time_s": mc_time,
                    "fd_time_s": fd_time,
                }
            )
    return pd.DataFrame(rows)


def build_variance_reduction_table() -> pd.DataFrame:
    """Compare MC estimator variance across antithetic/control-variate combinations for one ATM call."""
    K, option_type = SPOT, "call"
    rows = []
    for antithetic in [False, True]:
        for control_variate in [False, True]:
            result = mc_price(
                SPOT, K, MATURITY, RATE, SIGMA, option_type, DIV,
                n_paths=N_PATHS_TABLE, seed=SEED, antithetic=antithetic, control_variate=control_variate,
            )
            rows.append(
                {
                    "antithetic": antithetic,
                    "control_variate": control_variate,
                    "price": result.price,
                    "variance": result.variance,
                    "std_error": result.std_error,
                }
            )
    return pd.DataFrame(rows)


def _mean_abs_error(option_type: str, K: float, bs: float, n: int, antithetic: bool, control_variate: bool, n_seeds: int) -> tuple[float, float]:
    """Mean and std of |MC price - BS price| over n_seeds independent runs at a fixed path count."""
    errors = [
        abs(
            mc_price(
                SPOT, K, MATURITY, RATE, SIGMA, option_type, DIV,
                n_paths=n, seed=seed, antithetic=antithetic, control_variate=control_variate,
            ).price
            - bs
        )
        for seed in range(n_seeds)
    ]
    return float(np.mean(errors)), float(np.std(errors))


def plot_mc_convergence(n_seeds: int = 10) -> None:
    """Plot mean |MC price - BS price| vs path count, raw vs variance-reduced, averaged over n_seeds runs per point.

    A single-seed error trajectory is signed and can pass near zero by chance at any path count,
    which makes a one-shot convergence plot noisy enough to obscure the actual O(1/sqrt(n)) trend.
    Averaging over independent seeds at each path count gives a trend that isn't an artifact of
    one particular random draw.
    """
    K, option_type = SPOT, "call"
    bs = bs_price(SPOT, K, MATURITY, RATE, SIGMA, option_type, DIV)
    path_counts = np.unique(np.logspace(2, 6, 15).astype(int))

    raw = [_mean_abs_error(option_type, K, bs, int(n), False, False, n_seeds) for n in path_counts]
    reduced = [_mean_abs_error(option_type, K, bs, int(n), True, True, n_seeds) for n in path_counts]
    raw_mean, raw_std = (np.array(x) for x in zip(*raw))
    reduced_mean, reduced_std = (np.array(x) for x in zip(*reduced))

    reference = raw_mean[0] * np.sqrt(path_counts[0] / path_counts.astype(float))

    def _yerr(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return np.vstack([np.minimum(std, mean * 0.999), std])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(path_counts, raw_mean, yerr=_yerr(raw_mean, raw_std), fmt="o-", capsize=3, label="raw MC")
    ax.errorbar(
        path_counts, reduced_mean, yerr=_yerr(reduced_mean, reduced_std), fmt="o-", capsize=3,
        label="antithetic + control variate",
    )
    ax.plot(path_counts, reference, "k--", alpha=0.5, label=r"O(1/$\sqrt{n}$) reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of simulated paths")
    ax.set_ylabel(f"mean |MC price - BS price| ({n_seeds} seeds/point)")
    ax.set_title("Monte Carlo convergence: mean error vs path count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mc_convergence.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate all Phase 1 pricing deliverables: comparison table, variance-reduction table, convergence plot."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = build_pricing_comparison_table()
    comparison.to_csv(TABLES_DIR / "pricing_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    variance = build_variance_reduction_table()
    variance.to_csv(TABLES_DIR / "mc_variance_reduction.csv", index=False)
    print(variance.to_string(index=False))

    plot_mc_convergence()
    print(f"Saved deliverables to {TABLES_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
