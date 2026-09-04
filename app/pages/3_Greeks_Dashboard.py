"""Greeks Dashboard: analytical vs. finite-difference vs. Monte Carlo Greeks, side by side."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dpre.greeks import analytical, finite_difference
from dpre.greeks.monte_carlo import likelihood_ratio_delta, likelihood_ratio_vega, pathwise_delta, pathwise_vega
from sidebar import DIVIDEND_YIELD_BOUNDS, SIGMA_BOUNDS, render_prefill_button, seed_defaults

st.set_page_config(page_title="Greeks Dashboard", layout="wide")
st.title("Greeks Dashboard")
st.caption("dpre.greeks.{analytical,finite_difference,monte_carlo} — same code as scripts/03_greeks_comparison.py")

with st.sidebar:
    st.header("Option parameters")
    seed_defaults({"S_input": 100.0, "K_input": 100.0, "q_input": 0.01, "sigma_input": 0.20})
    render_prefill_button("S_input", "sigma_input", "q_input", k_key="K_input")
    # No `value=` on the four prefillable widgets below: seed_defaults already seeded them, and
    # passing both would warn about the redundant default (see sidebar.seed_defaults).
    S = st.number_input("Spot (S)", min_value=0.01, step=1.0, key="S_input")
    K = st.number_input("Strike (K)", min_value=0.01, step=1.0, key="K_input")
    T = st.slider("Maturity T (years)", 0.01, 3.0, 1.0, step=0.01)
    r = st.slider("Risk-free rate r", -0.02, 0.15, 0.03, step=0.005, format="%.3f")
    q = st.slider("Dividend yield q", *DIVIDEND_YIELD_BOUNDS, step=0.005, format="%.3f", key="q_input")
    sigma = st.slider("Volatility (sigma)", *SIGMA_BOUNDS, step=0.01, key="sigma_input")
    option_type = st.radio("Option type", ["call", "put"], horizontal=True)
    n_paths_mc = st.select_slider("MC paths (delta/vega)", options=[10**e for e in range(3, 7)], value=100_000, format_func=lambda n: f"{n:,}")

analytical_values = {
    "delta": analytical.delta(S, K, T, r, sigma, option_type, q),
    "gamma": analytical.gamma(S, K, T, r, sigma, q),
    "vega": analytical.vega(S, K, T, r, sigma, q),
    "theta": analytical.theta(S, K, T, r, sigma, option_type, q),
    "rho": analytical.rho(S, K, T, r, sigma, option_type, q),
}
fd_values = {
    "delta": finite_difference.delta(S, K, T, r, sigma, option_type, q),
    "gamma": finite_difference.gamma(S, K, T, r, sigma, q),
    "vega": finite_difference.vega(S, K, T, r, sigma, q),
    "theta": finite_difference.theta(S, K, T, r, sigma, option_type, q),
    "rho": finite_difference.rho(S, K, T, r, sigma, option_type, q),
}

pw_delta = pathwise_delta(S, K, T, r, sigma, option_type, q, n_paths=n_paths_mc)
lr_delta = likelihood_ratio_delta(S, K, T, r, sigma, option_type, q, n_paths=n_paths_mc)
pw_vega = pathwise_vega(S, K, T, r, sigma, option_type, q, n_paths=n_paths_mc)
lr_vega = likelihood_ratio_vega(S, K, T, r, sigma, option_type, q, n_paths=n_paths_mc)

rows = []
for greek in ["delta", "gamma", "vega", "theta", "rho"]:
    row = {
        "Greek": greek,
        "Analytical": analytical_values[greek],
        "Finite difference": fd_values[greek],
        "FD error": abs(fd_values[greek] - analytical_values[greek]),
    }
    if greek == "delta":
        row["MC pathwise"] = pw_delta.value
        row["MC pathwise std err"] = pw_delta.std_error
        row["MC likelihood-ratio"] = lr_delta.value
        row["MC LR std err"] = lr_delta.std_error
    elif greek == "vega":
        row["MC pathwise"] = pw_vega.value
        row["MC pathwise std err"] = pw_vega.std_error
        row["MC likelihood-ratio"] = lr_vega.value
        row["MC LR std err"] = lr_vega.std_error
    rows.append(row)

st.dataframe(pd.DataFrame(rows).round(6), width='stretch', hide_index=True)

st.divider()
st.subheader("Delta and gamma across spot")
st.caption(f"Strike, maturity, rate, vol, dividend held fixed at the sidebar values; current spot (${S:.2f}) marked.")

S_range = np.linspace(max(0.4 * K, 1.0), 1.6 * K, 200)
delta_curve = analytical.delta(S_range, K, T, r, sigma, option_type, q)
gamma_curve = analytical.gamma(S_range, K, T, r, sigma, q)

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Scatter(x=S_range, y=delta_curve, mode="lines", name="delta"))
    fig.add_vline(x=S, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="spot", yaxis_title="delta", height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')
with col2:
    fig = go.Figure(go.Scatter(x=S_range, y=gamma_curve, mode="lines", name="gamma", line=dict(color="orange")))
    fig.add_vline(x=S, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="spot", yaxis_title="gamma", height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')
