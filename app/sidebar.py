"""Shared sidebar controls: ticker + dividend-yield selection, persisted across pages via session_state."""

import streamlit as st

from cache_utils import suggest_dividend_yield

DEFAULT_TICKER = "SPY"

# Shared with cache_utils.suggest_dividend_yield's own clamp range: every dividend-yield slider in
# the app must cover at least what that function can return, or a genuinely high-yield ticker (a
# REIT, a utility, ...) crashes the widget with StreamlitValueAboveMaxError the first time its
# auto-estimated yield is pre-set as that slider's value -- SPY's ~1.3% never triggers this, which
# is exactly why a narrower bound can look fine for a long time before a real ticker hits it.
DIVIDEND_YIELD_BOUNDS = (0.0, 0.15)
SIGMA_BOUNDS = (0.01, 3.0)

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
            div_key = f"div_yield_{ticker}"
            if div_key not in st.session_state:  # seed once; passing value= as well would warn once overridden
                # Clamped defensively rather than trusting suggest_dividend_yield's own internal
                # clamp to stay forever in sync with this slider's bounds -- that assumption is
                # exactly what caused the original bug (two places, one changed without the other).
                st.session_state[div_key] = max(DIVIDEND_YIELD_BOUNDS[0], min(DIVIDEND_YIELD_BOUNDS[1], float(suggested)))
            dividend_yield = st.slider(
                "Dividend yield", *DIVIDEND_YIELD_BOUNDS, step=0.001, format="%.3f",
                key=div_key,
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

    sigma/q are clamped to SIGMA_BOUNDS/DIVIDEND_YIELD_BOUNDS before being written to
    session_state: the calling pages' own sliders use those same bounds, and a live value outside
    them (a high-yield ticker, an illiquid/volatile one whose near-term ATM vol runs high) would
    otherwise crash the widget with StreamlitValueAboveMaxError the moment it's next drawn, since
    Streamlit validates whatever's in session_state against the widget's declared range regardless
    of how it got there.
    """
    from cache_utils import DEFAULT_MAX_EXPIRIES, get_calibration, suggest_dividend_yield
    from dpre.calibration.svi import svi_implied_vol

    def _clamp(val: float, bounds: tuple[float, float]) -> float:
        return max(bounds[0], min(bounds[1], val))

    ticker = st.session_state.get("ticker", DEFAULT_TICKER)
    if st.button(f"Use live {ticker} spot & near-term vol", help="Fetches this page's own copy of the ticker's calibration; other pages are unaffected."):
        try:
            dividend_yield = suggest_dividend_yield(ticker)
            snapshot, _, svi_slices = get_calibration(ticker, dividend_yield, DEFAULT_MAX_EXPIRIES)
            atm_slice = min(svi_slices, key=lambda p: abs(p.T - 30 / 365))
            live_sigma = float(svi_implied_vol(0.0, atm_slice))
            st.session_state[s_key] = max(0.01, round(snapshot.spot, 2))
            st.session_state[sigma_key] = round(_clamp(live_sigma, SIGMA_BOUNDS), 4)
            st.session_state[q_key] = round(_clamp(snapshot.dividend_yield, DIVIDEND_YIELD_BOUNDS), 4)
            if k_key:
                st.session_state[k_key] = max(0.01, round(snapshot.spot, 2))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not fetch live data for {ticker}: {exc}")
            return
        st.rerun()
