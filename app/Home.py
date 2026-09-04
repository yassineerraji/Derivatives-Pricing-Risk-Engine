"""Landing page for the Derivatives Pricing & Risk Engine explorer app."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from cache_utils import DEFAULT_MAX_EXPIRIES, get_calibration
from dpre.calibration.arbitrage import check_butterfly_arbitrage, check_calendar_arbitrage
from sidebar import render_ticker_selector

st.set_page_config(page_title="Derivatives Pricing & Risk Engine", layout="wide")

st.title("Derivatives Pricing & Risk Engine")
st.markdown(
    "This is a quantitative finance engine built from first principles such as closed-form and numerical "
    "pricing, arbitrage-constrained volatility surface calibration, Greeks by three independent "
    "methods, and a full risk desk workflow (VaR/ES, delta-hedging cost). " \
    "This app is driven by **live market data for any ticker chosen**."
)
st.caption(
   "Have fun exploring the app !"
)

ticker, dividend_yield = render_ticker_selector()

st.divider()
st.subheader("Live snapshot")

try:
    snapshot, iv_df, svi_slices = get_calibration(ticker, dividend_yield, DEFAULT_MAX_EXPIRIES)

    butterfly_results = [check_butterfly_arbitrage(p) for p in svi_slices]
    n_butterfly_ok = sum(b.arbitrage_free for b in butterfly_results)
    calendar = check_calendar_arbitrage(svi_slices, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield)
    n_solved = len(iv_df)
    n_quoted = len(snapshot.chain)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(ticker, f"${snapshot.spot:,.2f}")
    m2.metric("Risk-free rate", f"{snapshot.risk_free_rate:.2%}")
    m3.metric("Dividend yield", f"{snapshot.dividend_yield:.2%}")
    m4.metric("Quotes solved", f"{n_solved}/{n_quoted}", f"{n_solved / n_quoted:.0%}")
    m5.metric("SVI maturities", len(svi_slices), f"{svi_slices[0].T * 365:.0f}-{svi_slices[-1].T * 365:.0f}d")

    badge_col1, badge_col2 = st.columns(2)
    with badge_col1:
        if n_butterfly_ok == len(svi_slices):
            st.success(f"Butterfly-arbitrage-free: {n_butterfly_ok}/{len(svi_slices)} slices")
        else:
            st.warning(f"Butterfly-arbitrage-free: {n_butterfly_ok}/{len(svi_slices)} slices")
    with badge_col2:
        if calendar.arbitrage_free:
            st.success("Calendar-arbitrage-free across all consecutive maturities")
        else:
            st.warning(f"Calendar arbitrage in {len(calendar.violations)} maturity pair(s)")

    st.caption(
        f"As of {snapshot.as_of}. The SVI surface is fit live, in increasing maturity order, "
        "each slice constrained (SLSQP) against Durrleman's no-arbitrage condition and against the "
        "previous maturity — see the Vol Surface Explorer for the sandbox that shows why that's needed."
    )
except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed (no silent fallback)
    st.error(f"Could not fetch/calibrate live data for {ticker}: {exc}")

st.divider()
st.subheader("Implemented from scratch")
st.caption("NumPy/SciPy only. QuantLib appears solely in `tests/` to cross-validate results, never in the pricing/risk code itself.")

st.markdown(
    """
| Module | Methods | Demonstrates |
| --- | --- | --- |
| **Pricing** | Closed-form Black-Scholes · Monte Carlo (antithetic + control variate) · Crank-Nicolson finite differences | Numerical methods validated against each other and a closed-form baseline |
| **Calibration** | Brent / Newton-Raphson implied vol · SVI smile parameterization · SLSQP-constrained arbitrage-free surface fitting | Market-data-aware calibration, not textbook curve fitting |
| **Greeks** | Analytical · bump-and-reprice finite differences · pathwise & likelihood-ratio Monte Carlo | Sensitivity analysis cross-checked across independent methods |
| **Risk** | Historical & Monte Carlo VaR/ES with bootstrap confidence intervals · multi-path delta-hedging with transaction costs | A practical risk-desk workflow, not a single static number |
"""
)

st.divider()
st.subheader("Explore")

pages = [
    ("pages/1_Pricing_Lab.py", "Pricing Lab", "Compare Black-Scholes, Monte Carlo, and finite differences live; watch variance reduction and MC convergence respond as you change path count."),
    ("pages/2_Vol_Surface_Explorer.py", "Vol Surface Explorer", "The live, arbitrage-constrained SVI surface for your chosen ticker; drag a slice's own parameters and watch the no-arbitrage condition respond in real time."),
    ("pages/3_Greeks_Dashboard.py", "Greeks Dashboard", "Analytical vs. finite-difference vs. Monte Carlo Greeks side by side, with delta/gamma curves across spot.                                         "),
    ("pages/4_Risk_Desk.py", "Risk Desk", "Edit an option book, compute historical and Monte Carlo VaR/ES, and run a multi-path hedging simulation to see what transaction costs really cost."),
]
cols = st.columns(4)
for col, (path, label, desc) in zip(cols, pages):
    with col:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(desc)
            st.page_link(path, label="Open")

st.divider()
st.caption(
    "Python · NumPy · SciPy · pandas · yfinance · Streamlit · Plotly. "
    "[Source on GitHub](https://github.com/yassineerraji/Derivatives-Pricing-Risk-Engine). "
    "Educational project — nothing here is investment advice."
)
