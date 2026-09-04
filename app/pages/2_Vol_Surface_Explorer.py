"""Vol Surface Explorer: the live calibrated SVI surface, plus a hands-on single-slice arbitrage sandbox."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cache_utils import DEFAULT_MAX_EXPIRIES, get_calibration
from dpre.calibration.arbitrage import check_butterfly_arbitrage, check_calendar_arbitrage
from dpre.calibration.svi import SVIParams, durrleman_g, svi_implied_vol
from sidebar import render_ticker_selector

st.set_page_config(page_title="Vol Surface Explorer", layout="wide")
st.title("Vol Surface Explorer")
st.caption("dpre.calibration.{data,implied_vol,svi,arbitrage} — same pipeline as scripts/02_calibrate_surface.py")

ticker, dividend_yield = render_ticker_selector()

try:
    snapshot, iv_df, svi_slices = get_calibration(ticker, dividend_yield, DEFAULT_MAX_EXPIRIES)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not fetch/calibrate live data for {ticker}: {exc}")
    st.stop()

st.subheader(f"{snapshot.ticker} implied volatility surface (spot ${snapshot.spot:,.2f})")

fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=iv_df["strike"], y=iv_df["T"] * 365, z=iv_df["implied_vol"],
        mode="markers", marker=dict(size=2, opacity=0.4), name="market IV",
    )
)
for params in svi_slices:
    slice_df = iv_df[iv_df["T"] == params.T]
    if slice_df.empty:
        continue
    forward = snapshot.spot * np.exp((snapshot.risk_free_rate - snapshot.dividend_yield) * params.T)
    strikes = np.linspace(slice_df["strike"].min(), slice_df["strike"].max(), 60)
    fitted_iv = svi_implied_vol(np.log(strikes / forward), params)
    fig.add_trace(
        go.Scatter3d(
            x=strikes, y=np.full_like(strikes, params.T * 365), z=fitted_iv,
            mode="lines", line=dict(color="orange", width=4), name=f"SVI fit ({params.T * 365:.0f}d)",
            showlegend=False,
        )
    )
fig.update_layout(
    scene=dict(xaxis_title="strike", yaxis_title="maturity (days)", zaxis_title="implied vol"),
    height=650, margin=dict(t=10, b=10),
)
st.plotly_chart(fig, width='stretch')

st.divider()
st.subheader("No-arbitrage checks")
st.caption("Durrleman's condition (butterfly, per slice) and total variance non-decreasing in T (calendar, across slices).")

butterfly_rows = [check_butterfly_arbitrage(p) for p in svi_slices]
calendar = check_calendar_arbitrage(svi_slices, snapshot.spot, snapshot.risk_free_rate, snapshot.dividend_yield)
arb_table = pd.DataFrame(
    {
        "maturity (days)": [round(p.T * 365) for p in svi_slices],
        "butterfly arbitrage-free": [b.arbitrage_free for b in butterfly_rows],
        "min g(k)": [round(b.min_g, 5) for b in butterfly_rows],
    }
)
st.dataframe(arb_table, width='stretch', hide_index=True)
if calendar.arbitrage_free:
    st.success("Calendar arbitrage-free across all consecutive maturity pairs.")
else:
    st.error(f"Calendar arbitrage found in {len(calendar.violations)} maturity pair(s).")

st.divider()
st.subheader("Single-slice arbitrage sandbox")
st.markdown(
    "Pick a calibrated maturity, then drag its SVI parameters away from the fitted values and watch "
    "**Durrleman's g(k)** respond — where it dips below zero is exactly where the smile implies a "
    "negative risk-neutral density (butterfly arbitrage)."
)

maturity_labels = {f"{p.T * 365:.0f} days (T={p.T:.3f})": p for p in svi_slices}
chosen_label = st.selectbox("Maturity", list(maturity_labels.keys()), key=f"vs_maturity_{ticker}")
base = maturity_labels[chosen_label]

# Sliders are keyed by (ticker, maturity): Streamlit only applies `value=` the first time a key is
# created, so an unkeyed slider would silently keep a stale number from a previous ticker/maturity
# instead of resetting to the new slice's own calibrated parameters.
#
# Bounds must cover whatever calibrate_svi_slice_arbitrage_free's own optimizer bounds allow (see
# svi.py), not an arbitrary "reasonable-looking" range: b's optimizer lower bound is 1e-8, and a
# real slice for a real ticker has landed there (a near-flat wing) -- a narrower UI range would
# crash on a calibrated value it can't display, not just on a user dragging past the edge.
BOUNDS = {"a": (-1.0, 1.0), "b": (0.0, 2.0), "rho": (-0.999, 0.999), "m": (-2.0, 2.0), "sigma": (0.0, 2.0)}

slice_key = f"{ticker}_{chosen_label}"
a_key, b_key, rho_key, m_key, sigma_key = (f"svi_{p}_{slice_key}" for p in ["a", "b", "rho", "m", "sigma"])
defaults = {a_key: base.a, b_key: base.b, rho_key: base.rho, m_key: base.m, sigma_key: base.sigma}
param_of_key = {a_key: "a", b_key: "b", rho_key: "rho", m_key: "m", sigma_key: "sigma"}


def _clamped(key: str, val: float) -> float:
    lo, hi = BOUNDS[param_of_key[key]]
    return max(lo, min(hi, float(val)))


for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = _clamped(key, val)


def _reset_svi_sliders() -> None:
    """on_click callback: Streamlit runs this before the sliders below are re-instantiated on the
    next rerun, which is the only safe time to overwrite a widget's session_state by key -- setting
    it after the widget already exists in the current run raises StreamlitWidgetAlreadyInstantiatedError."""
    for key, val in defaults.items():
        st.session_state[key] = _clamped(key, val)


slider_col, plot_col = st.columns([1, 2])
with slider_col:
    # No `value=` here: session_state is pre-seeded above, and passing both raises a Streamlit
    # warning about the widget's default being redundant with its already-set session_state.
    a = st.slider("a (level)", *BOUNDS["a"], step=0.001, format="%.3f", key=a_key)
    b = st.slider("b (angle/wing steepness)", *BOUNDS["b"], step=0.001, format="%.3f", key=b_key)
    rho = st.slider("rho (skew)", *BOUNDS["rho"], step=0.001, format="%.3f", key=rho_key)
    m = st.slider("m (horizontal shift)", *BOUNDS["m"], step=0.001, format="%.3f", key=m_key)
    sigma_svi = st.slider("sigma (curvature at the money)", *BOUNDS["sigma"], step=0.001, format="%.3f", key=sigma_key)
    st.button("Reset to calibrated values", on_click=_reset_svi_sliders)
    if any(abs(_clamped(key, val) - float(val)) > 1e-9 for key, val in defaults.items()):
        st.caption("One or more of this slice's calibrated values fell outside the slider range and were clamped for display.")

    custom = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma_svi, T=base.T)
    result = check_butterfly_arbitrage(custom)
    if result.arbitrage_free:
        st.success(f"Butterfly arbitrage-free (min g(k) = {result.min_g:.5f})")
    else:
        st.error(f"Butterfly arbitrage present (min g(k) = {result.min_g:.5f})")

with plot_col:
    k_grid = np.linspace(-1.2, 1.2, 300)
    forward = snapshot.spot * np.exp((snapshot.risk_free_rate - snapshot.dividend_yield) * base.T)
    slice_df = iv_df[iv_df["T"] == base.T]
    k_market = np.log(slice_df["strike"].to_numpy() / forward) if not slice_df.empty else np.array([])

    smile_fig = go.Figure()
    smile_fig.add_trace(go.Scatter(x=k_grid, y=svi_implied_vol(k_grid, base), mode="lines", name="calibrated (fixed)", line=dict(dash="dash", color="gray")))
    smile_fig.add_trace(go.Scatter(x=k_grid, y=svi_implied_vol(k_grid, custom), mode="lines", name="your parameters", line=dict(color="orange")))
    if len(k_market):
        smile_fig.add_trace(go.Scatter(x=k_market, y=slice_df["implied_vol"], mode="markers", name="market IV", marker=dict(size=5, opacity=0.5)))
    smile_fig.update_layout(xaxis_title="log-moneyness k = ln(K/F)", yaxis_title="implied vol", height=280, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.2))
    st.plotly_chart(smile_fig, width='stretch')

    g = durrleman_g(k_grid, a, b, rho, m, sigma_svi)
    g_fig = go.Figure()
    g_fig.add_trace(go.Scatter(x=k_grid, y=g, mode="lines", name="g(k)", line=dict(color="orange")))
    g_fig.add_hline(y=0.0, line_color="black", line_width=1)
    bad = g < 0
    if bad.any():
        g_fig.add_trace(go.Scatter(x=k_grid[bad], y=g[bad], mode="markers", name="g(k) < 0", marker=dict(color="red", size=4)))
    g_fig.update_layout(xaxis_title="log-moneyness k", yaxis_title="Durrleman g(k)", height=280, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.2))
    st.plotly_chart(g_fig, width='stretch')
