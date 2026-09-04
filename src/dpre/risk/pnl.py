"""P&L attribution: frictionless Black-Scholes replication vs replication including transaction costs."""

from dataclasses import dataclass

import numpy as np

from dpre.calibration.svi import SVIParams
from dpre.risk.book import Position
from dpre.risk.hedging import HedgeCostModel, run_hedge_simulation, simulate_price_path
from dpre.risk.valuation import book_representative_vol


@dataclass
class PnLAttribution:
    """Cumulative P&L of the book+hedge over the simulation horizon across n_paths independent price
    paths, frictionless vs frictional. Arrays are shaped (n_paths, n_days+1) except `days`.

    A single path can look better or worse than typical purely by chance -- the same lesson Phase
    1's multi-seed MC convergence plot demonstrated for pricing. Running many independent paths and
    reporting their distribution (not one arbitrarily lucky or unlucky realization) is what actually
    answers "how much does hedging cost, typically."
    """

    days: np.ndarray
    price_paths: np.ndarray
    frictionless_pnl: np.ndarray
    frictional_pnl: np.ndarray
    cumulative_cost: np.ndarray


def run_pnl_attribution(
    positions: list[Position], S0: float, r: float, q: float, svi_slices: list[SVIParams],
    cost_model: HedgeCostModel, n_days: int = 50, dt_years: float = 1 / 365,
    n_paths: int = 200, seed: int = 42,
) -> PnLAttribution:
    """Simulate n_paths independent price paths at the book's vega-weighted implied vol, hedge the
    book daily along EACH with and without transaction costs, and compare the resulting P&L distributions.

    Within a single path, using the SAME path (and the same vol for both simulating it and
    pricing/hedging off it) for the frictionless and frictional runs isolates transaction costs as
    the only difference between them -- the point of the comparison. The simulation vol itself is
    the book's vega-weighted average implied vol (see valuation.book_representative_vol): the most
    defensible single number for a book spanning several strikes/maturities, though it does NOT make
    the frictionless baseline flat (see docs/technical_notes.md sec. 5 for why not, and why that's a
    real finding rather than something to fix away).
    """
    sigma = book_representative_vol(positions, S0, r, q, svi_slices)

    rng = np.random.default_rng(seed)
    path_seeds = rng.integers(0, 2**31 - 1, size=n_paths)

    n_steps = n_days + 1
    price_paths = np.empty((n_paths, n_steps))
    frictionless_pnl = np.empty((n_paths, n_steps))
    frictional_pnl = np.empty((n_paths, n_steps))
    cumulative_cost = np.empty((n_paths, n_steps))

    for i, path_seed in enumerate(path_seeds):
        path = simulate_price_path(S0, r, q, sigma, n_days, dt_years, int(path_seed))
        frictionless = run_hedge_simulation(positions, path, r, q, svi_slices, dt_years, cost_model=None)
        frictional = run_hedge_simulation(positions, path, r, q, svi_slices, dt_years, cost_model=cost_model)
        price_paths[i] = path
        frictionless_pnl[i] = frictionless["total_pnl"]
        frictional_pnl[i] = frictional["total_pnl"]
        cumulative_cost[i] = frictional["cumulative_cost"]

    return PnLAttribution(
        days=np.arange(n_steps),
        price_paths=price_paths,
        frictionless_pnl=frictionless_pnl,
        frictional_pnl=frictional_pnl,
        cumulative_cost=cumulative_cost,
    )
