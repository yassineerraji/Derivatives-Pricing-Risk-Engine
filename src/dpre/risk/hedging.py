"""Discrete delta-hedging simulation for an option book, with an optional bid-ask spread + linear market-impact cost."""

from dataclasses import dataclass

import numpy as np

from dpre.calibration.svi import SVIParams
from dpre.risk.book import Position
from dpre.risk.valuation import book_delta, book_value


@dataclass
class HedgeCostModel:
    """Linear transaction cost model, both terms relative to trade notional (shares traded * price):
    half_spread (crossing half the bid-ask spread) plus impact_coefficient * |shares traded| (linear
    market impact, i.e. cost grows with trade size, not just a flat spread)."""

    half_spread: float = 0.0
    impact_coefficient: float = 0.0

    def cost(self, shares_traded: float, price: float) -> float:
        """Dollar cost of trading `shares_traded` shares (signed or not; only magnitude matters) at `price`."""
        n = abs(shares_traded)
        return n * price * (self.half_spread + self.impact_coefficient * n)


def simulate_price_path(S0: float, r: float, q: float, sigma: float, n_days: int, dt_years: float, seed: int) -> np.ndarray:
    """One GBM price path of n_days+1 points (S0 included) at daily steps of dt_years, under the risk-neutral drift."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_days)
    log_returns = (r - q - 0.5 * sigma**2) * dt_years + sigma * np.sqrt(dt_years) * z
    log_path = np.log(S0) + np.concatenate([[0.0], np.cumsum(log_returns)])
    return np.exp(log_path)


def run_hedge_simulation(
    positions: list[Position], price_path: np.ndarray, r: float, q: float, svi_slices: list[SVIParams],
    dt_years: float, cost_model: HedgeCostModel | None = None,
) -> dict:
    """Daily-rebalanced delta hedge of the book along `price_path`.

    At each step the hedge is set to hedge_shares = -book_delta (delta-neutral). The hedge's cash
    account is self-financing (buying/selling shares is funded from cash, which also earns/pays the
    risk-free rate between rebalances) except for transaction costs, which are a real cash outflow.
    total_pnl tracks (option book + hedge shares + cash) mark-to-market, relative to day 0 — in a
    frictionless, continuously-rebalanced, vol-matched world this would be flat; discrete daily
    rebalancing alone already introduces small non-zero P&L (a well-known artifact), and
    frictional - frictionless P&L should equal minus the cumulative transaction cost exactly, since
    both runs trade identical share quantities (the cost model affects cash and cost_paid only, not
    the delta target that's traded).
    """
    cost_model = cost_model or HedgeCostModel()
    n_steps = len(price_path) - 1

    option_value = np.empty(n_steps + 1)
    hedge_shares = np.empty(n_steps + 1)
    cash = np.empty(n_steps + 1)
    cost_paid = np.zeros(n_steps + 1)

    option_value[0] = book_value(positions, price_path[0], r, q, svi_slices, elapsed_years=0.0)
    hedge_shares[0] = -book_delta(positions, price_path[0], r, q, svi_slices, elapsed_years=0.0)
    cost_paid[0] = cost_model.cost(hedge_shares[0], price_path[0])
    cash[0] = -hedge_shares[0] * price_path[0] - cost_paid[0]

    for t in range(1, n_steps + 1):
        elapsed_years = t * dt_years
        S = price_path[t]
        cash[t] = cash[t - 1] * np.exp(r * dt_years)
        option_value[t] = book_value(positions, S, r, q, svi_slices, elapsed_years=elapsed_years)
        target_shares = -book_delta(positions, S, r, q, svi_slices, elapsed_years=elapsed_years)
        traded = target_shares - hedge_shares[t - 1]
        cost_paid[t] = cost_model.cost(traded, S)
        cash[t] -= traded * S + cost_paid[t]
        hedge_shares[t] = target_shares

    hedge_value = hedge_shares * price_path
    portfolio = option_value + hedge_value + cash
    return {
        "option_value": option_value,
        "hedge_shares": hedge_shares,
        "hedge_value": hedge_value,
        "cash": cash,
        "cost_paid": cost_paid,
        "cumulative_cost": np.cumsum(cost_paid),
        "total_pnl": portfolio - portfolio[0],
    }
