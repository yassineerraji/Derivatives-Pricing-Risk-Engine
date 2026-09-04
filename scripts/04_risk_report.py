"""Deliverable: VaR/ES table (historical + Monte Carlo, with CIs) and the cumulative hedging P&L chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dpre.calibration.data import fetch_chain
from dpre.calibration.implied_vol import extract_iv_surface
from dpre.calibration.svi import calibrate_svi_surface
from dpre.risk.book import build_book
from dpre.risk.hedging import HedgeCostModel
from dpre.risk.pnl import PnLAttribution, run_pnl_attribution
from dpre.risk.var_es import VaRResult, historical_var_es, monte_carlo_var_es

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

TICKER = "SPY"
MAX_EXPIRIES = 8
CONFIDENCES = [0.95, 0.99]
COST_MODEL = HedgeCostModel(half_spread=0.0005, impact_coefficient=1e-6)  # 5bps half-spread + linear impact
HEDGE_HORIZON_DAYS = 50  # safely under the book's shortest (90-day) maturity


def _var_result_row(method: str, result: VaRResult) -> dict:
    return {
        "method": method,
        "confidence": result.confidence,
        "var": result.var,
        "var_ci_low": result.var_ci[0],
        "var_ci_high": result.var_ci[1],
        "es": result.es,
        "es_ci_low": result.es_ci[0],
        "es_ci_high": result.es_ci[1],
        "n_scenarios": result.n_scenarios,
    }


def build_var_es_table(book, spot: float, r: float, q: float, svi_slices) -> pd.DataFrame:
    """Historical and Monte Carlo VaR/ES at each confidence level, one row per (method, confidence)."""
    rows = []
    for confidence in CONFIDENCES:
        hist = historical_var_es(book, spot, r, q, svi_slices, ticker=TICKER, confidence=confidence)
        rows.append(_var_result_row("historical", hist))
        mc = monte_carlo_var_es(book, spot, r, q, svi_slices, confidence=confidence)
        rows.append(_var_result_row("monte_carlo", mc))
    return pd.DataFrame(rows)


def plot_hedging_pnl(attribution: PnLAttribution) -> None:
    """Cumulative hedging P&L across all simulated paths: mean +/- 10th-90th percentile band,
    frictionless vs frictional, plus -mean cumulative cost as a visual check they track.

    A single path's cumulative P&L is noisy enough (see docs/technical_notes.md sec. 5) that a
    one-path chart can look better or worse than typical by chance; the band makes that spread,
    and the fact that it's the frictional band that's shifted down, visible rather than implied.
    """
    n_paths = attribution.frictionless_pnl.shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))

    for pnl, label, color in [
        (attribution.frictionless_pnl, "frictionless (theoretical)", "C0"),
        (attribution.frictional_pnl, "frictional (with transaction costs)", "C1"),
    ]:
        mean = pnl.mean(axis=0)
        lo, hi = np.percentile(pnl, [10, 90], axis=0)
        ax.plot(attribution.days, mean, color=color, label=f"{label} (mean)")
        ax.fill_between(attribution.days, lo, hi, color=color, alpha=0.15, label=f"{label} (10th-90th pct)")

    ax.plot(
        attribution.days, -attribution.cumulative_cost.mean(axis=0), "--", color="gray", alpha=0.7,
        label="-mean cumulative transaction cost",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("trading day")
    ax.set_ylabel("cumulative hedging P&L ($)")
    ax.set_title(f"{TICKER} book: delta-hedge replication P&L, theoretical vs frictional ({n_paths} paths)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hedging_pnl.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Calibrate SPY, build the book, compute VaR/ES, run the hedging P&L attribution, save both deliverables."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = fetch_chain(TICKER, max_expiries=MAX_EXPIRIES)
    iv_df = extract_iv_surface(snapshot)
    svi_slices = calibrate_svi_surface(iv_df, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield)
    print(f"Calibrated {len(svi_slices)} SVI slices for {TICKER} (spot={snapshot.spot:.2f})")

    book = build_book(snapshot.spot)
    print(f"Book: {len(book)} positions, net quantity {sum(p.quantity for p in book):.0f}")

    var_es_table = build_var_es_table(book, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield, svi_slices)
    var_es_table.to_csv(TABLES_DIR / "var_es.csv", index=False)
    print(var_es_table.to_string(index=False))

    attribution = run_pnl_attribution(
        book, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield, svi_slices,
        COST_MODEL, n_days=HEDGE_HORIZON_DAYS,
    )
    plot_hedging_pnl(attribution)
    final_frictionless = attribution.frictionless_pnl[:, -1]
    final_frictional = attribution.frictional_pnl[:, -1]
    final_cost = attribution.cumulative_cost[:, -1]
    print(f"Final frictionless P&L: mean {final_frictionless.mean():.2f}, std {final_frictionless.std():.2f}")
    print(f"Final frictional P&L:   mean {final_frictional.mean():.2f}, std {final_frictional.std():.2f}")
    print(f"Total transaction cost: mean {final_cost.mean():.2f}, std {final_cost.std():.2f}")
    print(f"Saved deliverables to {TABLES_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
