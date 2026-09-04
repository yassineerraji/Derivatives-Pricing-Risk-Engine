"""Shared sidebar controls: ticker + dividend-yield selection, persisted across pages via session_state."""

import streamlit as st

from cache_utils import suggest_dividend_yield

DEFAULT_TICKER = "SPY"

CURATED_TICKERS = {
    "SPY": "S&P 500 ETF -- broad index, deep liquid chain, the project's original target",
    "QQQ": "Nasdaq-100 ETF -- tech-heavy index",
    "IWM": "Russell 2000 ETF -- small-cap index, typically higher vol",
    "AAPL": "Apple -- large-cap single name",
    "MSFT": "Microsoft -- large-cap single name",
    "NVDA": "Nvidia -- high-vol single name",
    "TSLA": "Tesla -- high-vol single name, pronounced skew",
}
CUSTOM_LABEL = "Custom ticker..."


def render_ticker_selector(show_dividend_override: bool = True) -> tuple[str, float]:
    """Render the ticker (+ optional dividend-yield override) picker in the sidebar.

    Persisted in st.session_state so the choice survives navigating between pages. Returns
    (ticker, dividend_yield); dividend_yield is auto-suggested per ticker (cache_utils.
    suggest_dividend_yield) and, when show_dividend_override is True, editable -- pricing every
    position off a fixed SPY-shaped assumption regardless of the chosen name would be a real
    correctness gap (a non-dividend-payer priced with SPY's ~1.3% yield misprices skew/parity).
    """
    with st.sidebar:
        st.subheader("Underlying")
        current = st.session_state.get("ticker", DEFAULT_TICKER)
        options = list(CURATED_TICKERS) + [CUSTOM_LABEL]
        default_index = options.index(current) if current in CURATED_TICKERS else len(options) - 1

        choice = st.selectbox(
            "Ticker", options, index=default_index,
            format_func=lambda t: t if t == CUSTOM_LABEL else f"{t} -- {CURATED_TICKERS[t]}",
        )
        if choice == CUSTOM_LABEL:
            ticker = st.text_input(
                "Symbol", value=current if current not in CURATED_TICKERS else "",
                placeholder="e.g. AMD",
            ).strip().upper()
            if not ticker:
                st.caption("Enter a symbol to continue; showing SPY until then.")
                ticker = DEFAULT_TICKER
        else:
            ticker = choice
        st.session_state.ticker = ticker

        suggested = suggest_dividend_yield(ticker)
        if show_dividend_override:
            dividend_yield = st.slider(
                "Dividend yield", 0.0, 0.08, float(suggested), step=0.001, format="%.3f",
                key=f"div_yield_{ticker}",
                help="Auto-estimated from Yahoo Finance; drag to override if you have a better estimate.",
            )
        else:
            dividend_yield = suggested
        st.session_state.dividend_yield = dividend_yield

    return ticker, dividend_yield


def seed_defaults(defaults: dict) -> None:
    """Pre-seed session_state for widget keys that don't have a value yet (first page load), so
    their widgets can be created without a `value=` argument -- passing both `value=` and a `key=`
    that already has a stored value (e.g. after render_prefill_button runs) triggers a Streamlit
    warning about the redundant default.
    """
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_prefill_button(s_key: str, sigma_key: str, q_key: str, k_key: str | None = None) -> None:
    """A sidebar button that prefills S / sigma / q (and optionally K, at the money) from the
    ticker currently selected elsewhere in the app (Home/Vol Surface Explorer/Risk Desk), using its
    live spot and near-term ATM SVI vol. For generic pages (Pricing Lab, Greeks Dashboard) that
    aren't tied to one underlying but are more useful with a real starting point than an arbitrary
    S=100, sigma=0.20 default.
    """
    from cache_utils import DEFAULT_MAX_EXPIRIES, get_calibration, suggest_dividend_yield
    from dpre.calibration.svi import svi_implied_vol

    ticker = st.session_state.get("ticker", DEFAULT_TICKER)
    if st.button(f"Use live {ticker} spot & near-term vol", help="Fetches this page's own copy of the ticker's calibration; other pages are unaffected."):
        try:
            dividend_yield = suggest_dividend_yield(ticker)
            snapshot, _, svi_slices = get_calibration(ticker, dividend_yield, DEFAULT_MAX_EXPIRIES)
            atm_slice = min(svi_slices, key=lambda p: abs(p.T - 30 / 365))
            st.session_state[s_key] = round(snapshot.spot, 2)
            st.session_state[sigma_key] = round(float(svi_implied_vol(0.0, atm_slice)), 4)
            st.session_state[q_key] = round(snapshot.dividend_yield, 4)
            if k_key:
                st.session_state[k_key] = round(snapshot.spot, 2)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not fetch live data for {ticker}: {exc}")
            return
        st.rerun()
