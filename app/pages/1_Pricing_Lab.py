"""Pricing Lab: live Black-Scholes / Monte Carlo / finite-difference comparison and MC variance reduction."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dpre.pricing.black_scholes import price as bs_price
from dpre.pricing.finite_difference import price_crank_nicolson
from dpre.pricing.monte_carlo import mc_price
from sidebar import DIVIDEND_YIELD_BOUNDS, SIGMA_BOUNDS, render_prefill_button, seed_defaults

st.set_page_config(page_title="Pricing Lab", layout="wide")
st.title("Pricing Lab")
st.caption("dpre.pricing.black_scholes / monte_carlo / finite_difference — same code as scripts/01_pricing_comparison.py")

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

    st.header("Monte Carlo settings")
    n_paths = st.select_slider(
        "Path count", options=[10**e for e in range(2, 7)], value=10_000,
        format_func=lambda n: f"{n:,}",
    )
    antithetic = st.checkbox("Antithetic variates", value=True)
    control_variate = st.checkbox("Control variate", value=True)
    seed = st.number_input("Random seed", value=42, step=1)

bs = bs_price(S, K, T, r, sigma, option_type, q)
mc = mc_price(S, K, T, r, sigma, option_type, q, n_paths=n_paths, seed=seed, antithetic=antithetic, control_variate=control_variate)
t0 = time.perf_counter()
fd = price_crank_nicolson(S, K, T, r, sigma, option_type, q)
fd_time = time.perf_counter() - t0

col1, col2, col3 = st.columns(3)
col1.metric("Black-Scholes (closed-form)", f"{bs:.4f}")
col2.metric("Monte Carlo", f"{mc.price:.4f}", delta=f"{mc.price - bs:+.4f} vs BS")
col3.metric("Crank-Nicolson FD", f"{fd:.4f}", delta=f"{fd - bs:+.4f} vs BS")

st.caption(
    f"MC 95% CI: [{mc.ci_low:.4f}, {mc.ci_high:.4f}] (contains BS: {mc.ci_low <= bs <= mc.ci_high}) "
    f"| MC std error: {mc.std_error:.5f} | FD runtime: {fd_time * 1000:.2f} ms"
)

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Variance reduction")
    st.caption("Same (S, K, T, r, sigma, q) and path count, all four antithetic/control-variate combinations.")

    @st.cache_data(show_spinner=False)
    def variance_reduction_table(S, K, T, r, sigma, option_type, q, n_paths, seed):
        rows = []
        for anti in [False, True]:
            for cv in [False, True]:
                res = mc_price(S, K, T, r, sigma, option_type, q, n_paths=n_paths, seed=seed, antithetic=anti, control_variate=cv)
                label = f"{'antithetic' if anti else 'raw'} + {'CV' if cv else 'no CV'}"
                rows.append((label, res.variance))
        return rows

    vr_rows = variance_reduction_table(S, K, T, r, sigma, option_type, q, n_paths, seed)
    fig = go.Figure(go.Bar(x=[r[0] for r in vr_rows], y=[r[1] for r in vr_rows]))
    fig.update_layout(yaxis_title="estimator variance", height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("MC convergence")
    st.caption("|MC price - BS price| vs path count, raw vs. your current antithetic/CV settings (single seed).")

    @st.cache_data(show_spinner=False)
    def convergence_curve(S, K, T, r, sigma, option_type, q, seed, max_n):
        path_counts = np.unique(np.logspace(2, np.log10(max_n), 12).astype(int))
        raw_err, chosen_err = [], []
        for n in path_counts:
            raw = mc_price(S, K, T, r, sigma, option_type, q, n_paths=int(n), seed=seed, antithetic=False, control_variate=False)
            chosen = mc_price(S, K, T, r, sigma, option_type, q, n_paths=int(n), seed=seed, antithetic=True, control_variate=True)
            raw_err.append(abs(raw.price - bs_price(S, K, T, r, sigma, option_type, q)))
            chosen_err.append(abs(chosen.price - bs_price(S, K, T, r, sigma, option_type, q)))
        return path_counts, raw_err, chosen_err

    path_counts, raw_err, chosen_err = convergence_curve(S, K, T, r, sigma, option_type, q, seed, max(n_paths, 10_000))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path_counts, y=raw_err, mode="lines+markers", name="raw MC"))
    fig.add_trace(go.Scatter(x=path_counts, y=chosen_err, mode="lines+markers", name="antithetic + CV"))
    fig.update_layout(
        xaxis_type="log", yaxis_type="log", xaxis_title="number of paths", yaxis_title="|MC - BS|",
        height=380, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, width='stretch')
