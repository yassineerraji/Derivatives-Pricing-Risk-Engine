"""A simplified option book: fixed positions across strikes/maturities/types on a single underlying."""

from dataclasses import dataclass

MATURITIES_YEARS = {"3m": 90 / 365, "6m": 180 / 365, "9m": 270 / 365}


@dataclass(frozen=True)
class Position:
    """One option position: type, strike, initial time-to-maturity T (years), and signed quantity (long > 0)."""

    option_type: str
    strike: float
    T: float
    quantity: float


def build_book(spot: float) -> list[Position]:
    """An 8-position book on one underlying: mixed calls/puts, moneyness, maturities, and long/short quantities.

    Strikes are rounded to the nearest $5 (a realistic SPY-like tick) around the given spot. All
    maturities are >= 90 days so a hedging simulation of a few months stays well clear of any
    position expiring mid-run (see risk/pnl.py).
    """
    def strike_at(pct: float) -> float:
        return round(spot * pct / 5) * 5

    T = MATURITIES_YEARS
    return [
        Position("call", strike_at(0.95), T["3m"], 500),
        Position("put", strike_at(0.95), T["3m"], -300),
        Position("call", strike_at(1.00), T["6m"], -600),
        Position("put", strike_at(1.00), T["6m"], 800),
        Position("call", strike_at(1.05), T["6m"], 400),
        Position("put", strike_at(0.90), T["9m"], -700),
        Position("call", strike_at(1.10), T["9m"], 1000),
        Position("put", strike_at(1.05), T["3m"], 450),
    ]
