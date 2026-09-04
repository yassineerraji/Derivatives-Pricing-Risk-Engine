"""Deliverable: calibrated SVI volatility surface from a live options chain, plus the surface plot."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from dpre.calibration.arbitrage import check_butterfly_arbitrage, check_calendar_arbitrage
from dpre.calibration.data import fetch_chain
from dpre.calibration.implied_vol import extract_iv_surface
from dpre.calibration.svi import DEFAULT_MAX_ABS_LOG_MONEYNESS, svi_implied_vol, calibrate_svi_surface

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"

TICKER = "SPY"
MAX_EXPIRIES = 8


def plot_vol_surface(iv_df: pd.DataFrame, svi_slices, snapshot) -> None:
    """3D scatter of market implied vols plus the fitted SVI surface across strike, maturity, and IV."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(iv_df["strike"], iv_df["T"], iv_df["implied_vol"], s=8, alpha=0.5, label="market IV")

    for params in svi_slices:
        slice_df = iv_df[iv_df["T"] == params.T]
        if slice_df.empty:
            continue
        forward = snapshot.spot * np.exp((snapshot.risk_free_rate - snapshot.dividend_yield) * params.T)
        lo = forward * np.exp(-DEFAULT_MAX_ABS_LOG_MONEYNESS)
        hi = forward * np.exp(DEFAULT_MAX_ABS_LOG_MONEYNESS)
        strikes = np.linspace(max(lo, slice_df["strike"].min()), min(hi, slice_df["strike"].max()), 60)
        k = np.log(strikes / forward)
        fitted_iv = svi_implied_vol(k, params)
        ax.plot(strikes, np.full_like(strikes, params.T), fitted_iv, color="C1", linewidth=1.5)

    ax.set_xlabel("strike")
    ax.set_ylabel("maturity (years)")
    ax.set_zlabel("implied vol")
    ax.set_title(f"{TICKER} implied volatility surface: market points vs SVI fit")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "vol_surface.png", dpi=150)
    plt.close(fig)


def run_arbitrage_checks(svi_slices, spot: float, r: float, q: float) -> pd.DataFrame:
    """Durrleman's condition per slice (butterfly) and a calendar-spread check across maturities."""
    butterfly = [check_butterfly_arbitrage(p) for p in svi_slices]
    calendar = check_calendar_arbitrage(svi_slices, spot, r, q)

    if not calendar.arbitrage_free:
        for v in calendar.violations:
            print(
                f"CALENDAR ARBITRAGE: T={v.T_prev:.4f} -> T={v.T_curr:.4f}, "
                f"{v.n_violations} strikes, worst variance gap {v.worst_gap:.6f}"
            )

    return pd.DataFrame(
        {
            "T": [b.T for b in butterfly],
            "butterfly_arbitrage_free": [b.arbitrage_free for b in butterfly],
            "butterfly_min_g": [b.min_g for b in butterfly],
            "calendar_arbitrage_free": calendar.arbitrage_free,
        }
    )


def main() -> None:
    """Fetch the chain, extract the IV surface, fit SVI per maturity, and save the table + plot."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = fetch_chain(TICKER, max_expiries=MAX_EXPIRIES)
    print(f"{snapshot.ticker} spot={snapshot.spot:.2f} r={snapshot.risk_free_rate:.4f} q={snapshot.dividend_yield:.4f} "
          f"({len(snapshot.chain)} quotes across {snapshot.chain['T'].nunique()} expiries)")

    iv_df = extract_iv_surface(snapshot)
    print(f"Solved implied vol for {len(iv_df)}/{len(snapshot.chain)} quotes")

    svi_slices = calibrate_svi_surface(iv_df, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield)
    svi_table = pd.DataFrame([vars(p) for p in svi_slices])
    svi_table.to_csv(TABLES_DIR / "svi_params.csv", index=False)
    print(svi_table.to_string(index=False))

    arbitrage_table = run_arbitrage_checks(svi_slices, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield)
    arbitrage_table.to_csv(TABLES_DIR / "svi_arbitrage_check.csv", index=False)
    print(arbitrage_table.to_string(index=False))

    plot_vol_surface(iv_df, svi_slices, snapshot)
    print(f"Saved deliverables to {TABLES_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
