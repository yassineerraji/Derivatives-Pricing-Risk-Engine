"""Risk Desk: edit an option book, compute VaR/ES, and run a multi-path delta-hedging cost simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cache_utils import DEFAULT_MAX_EXPIRIES, get_calibration
from dpre.risk.book import Position, build_book
from dpre.risk.hedging import HedgeCostModel
from dpre.risk.pnl import run_pnl_attribution
from dpre.risk.var_es import historical_var_es, monte_carlo_var_es
from sidebar import render_ticker_selector

st.set_page_config(page_title="Risk Desk", layout="wide")
st.title("Risk Desk")
st.caption("dpre.risk.{book,valuation,var_es,hedging,pnl} — same code as scripts/04_risk_report.py")

ticker, dividend_yield = render_ticker_selector()

try:
    snapshot, iv_df, svi_slices = get_calibration(ticker, dividend_yield, DEFAULT_MAX_EXPIRIES)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not fetch/calibrate live data for {ticker}: {exc}")
    st.stop()

S0, r, q = snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield

st.subheader("Option book")
st.caption("Edit, add, or remove rows. Maturity is in days from today. Switching tickers resets the book to defaults for the new spot (old strikes would be off-moneyness otherwise).")

if st.session_state.get("book_ticker") != ticker:
    default_book = build_book(S0)
    st.session_state.book_df = pd.DataFrame(
        {
            "option_type": [p.option_type for p in default_book],
            "strike": [p.strike for p in default_book],
            "maturity_days": [round(p.T * 365) for p in default_book],
            "quantity": [p.quantity for p in default_book],
        }
    )
    st.session_state.book_ticker = ticker
    st.session_state.pop("var_es_result", None)
    st.session_state.pop("pnl_result", None)

edited_df = st.data_editor(
    st.session_state.book_df,
    num_rows="dynamic",
    width='stretch',
    column_config={
        "option_type": st.column_config.SelectboxColumn(options=["call", "put"]),
        "strike": st.column_config.NumberColumn(min_value=0.01),
        "maturity_days": st.column_config.NumberColumn(min_value=1),
        "quantity": st.column_config.NumberColumn(),
    },
)
st.session_state.book_df = edited_df

book = [
    Position(row.option_type, float(row.strike), float(row.maturity_days) / 365.0, float(row.quantity))
    for row in edited_df.dropna().itertuples()
]

if not book:
    st.warning("Add at least one position to compute risk.")
    st.stop()

net_quantity = sum(p.quantity for p in book)
shortest_maturity_days = min(p.T for p in book) * 365
st.caption(f"{len(book)} positions, net quantity {net_quantity:.0f}, shortest maturity {shortest_maturity_days:.0f} days.")

st.divider()
st.subheader("Value at Risk / Expected Shortfall")

col1, col2 = st.columns(2)
confidence = col1.select_slider("Confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
horizon_days = col2.number_input("Horizon (days)", value=1, min_value=1, max_value=10, step=1)

if st.button("Compute VaR/ES", type="primary"):
    with st.spinner("Running historical and Monte Carlo VaR/ES..."):
        hist = historical_var_es(book, S0, r, q, svi_slices, ticker=ticker, confidence=confidence, horizon_days=horizon_days)
        mc = monte_carlo_var_es(book, S0, r, q, svi_slices, confidence=confidence, horizon_days=horizon_days, n_scenarios=8000)
        st.session_state.var_es_result = (hist, mc)

if "var_es_result" in st.session_state:
    hist, mc = st.session_state.var_es_result
    var_table = pd.DataFrame(
        [
            {"method": "historical", "VaR": hist.var, "VaR CI": hist.var_ci, "ES": hist.es, "ES CI": hist.es_ci},
            {"method": "monte_carlo", "VaR": mc.var, "VaR CI": mc.var_ci, "ES": mc.es, "ES CI": mc.es_ci},
        ]
    )
    st.dataframe(var_table.round(2), width='stretch', hide_index=True)

st.divider()
st.subheader("Delta-hedging cost simulation")
st.markdown(
    "Simulates many independent price paths at the book's vega-weighted implied vol, delta-hedges "
    "the book daily along each with and without transaction costs, and compares the resulting P&L "
    "distributions. A single path is noisy enough to be misleading on its own -- see "
    "`docs/technical_notes.md` sec. 5 -- so this always runs a full distribution."
)

max_days = max(int(shortest_maturity_days) - 10, 5)
c1, c2, c3, c4 = st.columns(4)
n_days = c1.slider("Horizon (trading days)", 5, max_days, min(50, max_days))
n_paths = c2.slider("Number of paths", 20, 500, 150, step=10)
half_spread_bps = c3.slider("Half bid-ask spread (bps)", 0.0, 20.0, 5.0, step=0.5)
impact_bps = c4.slider("Market impact coefficient (bps per share)", 0.0, 20.0, 1.0, step=0.5)

cost_model = HedgeCostModel(half_spread=half_spread_bps / 10_000, impact_coefficient=impact_bps / 10_000)

if st.button("Run hedging simulation", type="primary"):
    with st.spinner(f"Simulating {n_paths} paths x {n_days} days..."):
        st.session_state.pnl_result = run_pnl_attribution(book, S0, r, q, svi_slices, cost_model, n_days=n_days, n_paths=n_paths)

if "pnl_result" in st.session_state:
    attribution = st.session_state.pnl_result
    final_frictionless = attribution.frictionless_pnl[:, -1]
    final_frictional = attribution.frictional_pnl[:, -1]
    final_cost = attribution.cumulative_cost[:, -1]

    m1, m2, m3 = st.columns(3)
    m1.metric("Frictionless P&L (mean)", f"${final_frictionless.mean():,.0f}", f"std ${final_frictionless.std():,.0f}")
    m2.metric("Frictional P&L (mean)", f"${final_frictional.mean():,.0f}", f"std ${final_frictional.std():,.0f}")
    m3.metric("Transaction cost (mean)", f"${final_cost.mean():,.0f}", f"std ${final_cost.std():,.0f}")

    fig = go.Figure()
    for pnl, label, color in [(attribution.frictionless_pnl, "frictionless", "rgb(31,119,180)"), (attribution.frictional_pnl, "frictional", "rgb(255,127,14)")]:
        mean = pnl.mean(axis=0)
        lo, hi = np.percentile(pnl, [10, 90], axis=0)
        fig.add_trace(go.Scatter(x=attribution.days, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=attribution.days, y=lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)"), name=f"{label} 10th-90th pct"))
        fig.add_trace(go.Scatter(x=attribution.days, y=mean, mode="lines", line=dict(color=color, width=2), name=f"{label} (mean)"))
    fig.add_hline(y=0.0, line_color="black", line_width=1)
    fig.update_layout(xaxis_title="trading day", yaxis_title="cumulative hedging P&L ($)", height=500, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width='stretch')
