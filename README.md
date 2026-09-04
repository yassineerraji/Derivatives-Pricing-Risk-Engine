# Derivatives Pricing & Risk Engine

An options pricing and risk management engine that implements several numerical methods from first principles, calibrates to real market data, and measures the practical cost of hedging.

> **Project brief**
> Build the models from first principles, compare their accuracy and performance, and use market data to connect theory with a realistic risk workflow.

Pricing, calibration, Greeks, and risk models are implemented directly with NumPy/SciPy — no pricing library. QuantLib appears only in `tests/`, as a cross-validation dependency, never inside `src/`. Black-Scholes closed-form pricing is the validation baseline throughout: every other method is judged by how well and how fast it reproduces it. A 4-page written report (`report.pdf`, source in `report.tex`) summarizes the headline results across all four modules; see [Report](#report) below.

## Contents

- [Scope](#scope)
- [Repository layout](#repository-layout)
- [Modules](#modules)
- [Interactive app](#interactive-app)
- [Report](#report)
- [Getting started](#getting-started)
- [Conventions & assumptions](#conventions--assumptions)
- [Technology](#technology)
- [Documentation requirements](#documentation-requirements)

## Scope

The engine covers:

- European option pricing with three independent numerical approaches
- Implied volatility extraction and SVI volatility-surface calibration
- Greeks computed analytically, by finite differences, and by Monte Carlo
- Historical and model-based VaR, Expected Shortfall, and discrete delta hedging

The Black-Scholes implementation is the baseline for validation, not a substitute for the numerical methods being developed.

## Repository layout

```
src/dpre/
  pricing/
    black_scholes.py      # closed-form call/put, d1/d2, put-call parity check
    monte_carlo.py         # GBM path simulation, antithetic + control variate
    finite_difference.py   # Crank-Nicolson PDE solver
  calibration/
    data.py                 # yfinance chain fetch + local cache
    implied_vol.py           # Brent/Newton-Raphson IV solver
    svi.py                   # SVI parameterization + weighted least-squares fit
    arbitrage.py              # butterfly (Durrleman) and calendar no-arbitrage checks
  greeks/
    analytical.py           # closed-form Greeks
    finite_difference.py    # bump-and-reprice
    monte_carlo.py           # pathwise + likelihood-ratio estimators
  risk/
    book.py                  # option book construction (8 positions)
    var_es.py                # historical-simulation and MC VaR/ES
    hedging.py                # discrete delta-hedge simulation with costs
    pnl.py                    # frictionless vs frictional P&L attribution
    valuation.py               # prices book positions off the calibrated SVI surface

scripts/                    # one entrypoint per deliverable, thin orchestration only
  01_pricing_comparison.py
  02_calibrate_surface.py
  03_greeks_comparison.py
  04_risk_report.py

tests/                      # pytest, mirrors src/dpre structure
data/cache/                 # gitignored raw yfinance pulls (parquet/csv, timestamped)
results/
  tables/                    # CSV outputs
  plots/                     # PNG/HTML outputs
docs/
  technical_notes.md         # model assumptions, smile discussion, variance-reduction proof, hedging-cost interpretation

app/                         # Streamlit UI over src/dpre
  Home.py
  cache_utils.py              # st.cache_data wrappers around the live fetch+calibrate pipeline
  sidebar.py                   # shared ticker/dividend-yield selector + generic-page prefill button
  pages/
    1_Pricing_Lab.py
    2_Vol_Surface_Explorer.py
    3_Greeks_Dashboard.py
    4_Risk_Desk.py

report.tex, report.pdf       # the written report (see Report below)
```

Scripts contain no pricing/math logic themselves — they call `src/dpre`, write to `results/`, and are what "the deliverable" means operationally. `app/` follows the same rule: pages call `src/dpre` and render results, no pricing/risk logic lives in `app/` itself.

## Modules

### 1. Pricing

Compare three pricing methods using price, runtime, and error against the closed-form Black-Scholes result.

| Method | Implementation | Purpose |
| --- | --- | --- |
| **Black-Scholes** | Closed-form European call and put | Validation reference |
| **Monte Carlo** | Risk-neutral GBM with antithetic and control variates | Quantify convergence and variance reduction |
| **Finite differences** | Crank-Nicolson scheme for the Black-Scholes PDE | Compare PDE accuracy and runtime with Monte Carlo |

The Monte Carlo analysis includes confidence intervals, variance before and after reduction, and a convergence plot of error against the number of simulations. The finite-difference comparison uses a common tolerance level.

**Deliverable:** `results/tables/pricing_comparison.csv`, `results/tables/mc_variance_reduction.csv`, `results/plots/mc_convergence.png` — produced by `scripts/01_pricing_comparison.py`.

### 2. Calibration to real data

1. Pull an options chain from Yahoo Finance (`yfinance`), using a liquid underlying (default SPY).
2. Extract implied volatility for each strike and maturity with Brent root-finding.
3. Calibrate an SVI (Stochastic Volatility Inspired) model to parameterize the volatility smile, per maturity slice, subject to nonlinear no-arbitrage constraints (Durrleman's butterfly condition and calendar monotonicity).
4. Visualize the resulting three-dimensional volatility surface across strike, maturity, and implied volatility.

**Deliverable:** `results/tables/svi_params.csv`, `results/tables/svi_arbitrage_check.csv`, `results/plots/vol_surface.png` — produced by `scripts/02_calibrate_surface.py`.

### 3. Greeks

Compare the following approaches for delta, gamma, vega, theta, and rho where applicable:

- **Analytical:** Closed-form Black-Scholes formulas
- **Finite differences:** Bump-and-reprice with a documented step size $h$ and a discussion of bias versus variance
- **Monte Carlo:** Pathwise and likelihood-ratio estimators, with delta and vega implemented at minimum

**Deliverable:** `results/tables/greeks_comparison.csv` — produced by `scripts/03_greeks_comparison.py`.

### 4. Risk management

An 8-position book across calls/puts, strikes, and maturities (3m/6m/9m) on a single underlying, priced off the calibrated SVI surface (sticky-strike) rather than one flat volatility.

- **VaR and Expected Shortfall:** Historical-simulation estimates from real returns and Monte Carlo estimates from the calibrated model, both with confidence intervals.
- **Discrete delta hedging:** Daily-rebalanced simulation with bid-ask spread and linear market-impact costs, across multiple independent price paths.
- **P&L attribution:** Frictionless Black-Scholes/SVI replication P&L versus P&L including transaction costs, isolating the real cost of hedging from smile-mismatch and discretization noise.

**Deliverable:** `results/plots/hedging_pnl.png`, `results/tables/var_es.csv` — produced by `scripts/04_risk_report.py`.

## Interactive app

A [Streamlit](https://streamlit.io) app under `app/` puts the four modules above behind sliders
instead of fixed script parameters — it calls `src/dpre` directly (no separate implementation) and
is covered by a smoke-test pass with `streamlit.testing.v1.AppTest`. Run it with:

```
uv run streamlit run app/Home.py
```

**Any ticker, not just SPY.** A shared sidebar control (a curated liquid-name dropdown plus a
free-text option) picks the underlying for Home, Vol Surface Explorer, and Risk Desk, persisted
across navigation; the two generic calculator pages (Pricing Lab, Greeks Dashboard) offer an
optional one-click prefill from whichever ticker is currently selected instead. Dividend yield is
auto-estimated per ticker and always shown as an editable override, not silently assumed.

Four pages, each captioned with the `dpre` modules it calls:

- **Pricing Lab** — live Black-Scholes/Monte Carlo/finite-difference comparison; drag path count and
  toggle antithetic/control-variate to watch variance reduction and MC convergence respond.
- **Vol Surface Explorer** — the live, arbitrage-constrained SVI surface calibrated to real market
  quotes, plus a single-slice sandbox where dragging `a, b, rho, m, sigma` redraws Durrleman's
  no-arbitrage condition `g(k)` in real time.
- **Greeks Dashboard** — analytical vs. finite-difference vs. Monte Carlo (pathwise / likelihood-ratio)
  Greeks side by side, with delta/gamma curves across spot.
- **Risk Desk** — an editable option book, live historical/Monte Carlo VaR/ES, and a multi-path
  delta-hedging simulation with adjustable spread/impact costs.

### Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io) (free, built for exactly this,
auto-redeploys on every push to `main`): repo `yassineerraji/Derivatives-Pricing-Risk-Engine`,
branch `main`, main file `app/Home.py`.

`requirements.txt` at the repo root is the install manifest Streamlit Cloud actually uses — it's
generated from `uv.lock` (`uv export --no-dev --no-hashes --format requirements-txt -o requirements.txt`)
so it can't silently drift from what's actually tested locally, and dev-only dependencies (QuantLib,
used solely for cross-validation in `tests/`) are correctly excluded from it, keeping the deployed
image lighter. Regenerate it with that same command any time `pyproject.toml` dependencies change.

## Report

`report.pdf` (LaTeX source: `report.tex`, compiled with [Tectonic](https://tectonic-typesetting.github.io))
is a 4-page written report summarizing the headline result of each module: pricing agreement across
methods, the measured Monte Carlo variance reduction, the SVI arbitrage-check failure-and-fix story,
Greek-estimator accuracy, and the isolated transaction-cost gap in the hedging simulation. It embeds
figures directly from `results/plots/`, so regenerate those first (see below) before recompiling with
`tectonic report.tex`.

## Getting started

Requires Python ≥3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# install dependencies
uv sync

# run the test suite
uv run pytest

# reproduce every deliverable (each writes to results/tables/ and results/plots/)
uv run python scripts/01_pricing_comparison.py
uv run python scripts/02_calibrate_surface.py
uv run python scripts/03_greeks_comparison.py
uv run python scripts/04_risk_report.py

# launch the interactive app
uv run streamlit run app/Home.py
```

Scripts 2 and 4 pull a live options chain via `yfinance`; a prior successful pull is cached under
`data/cache/` and reused with a printed warning if the live fetch fails, rather than fabricating data.

## Conventions & assumptions

- **Day count / compounding:** ACT/365, continuously-compounded rates, annualized volatility —
  applied uniformly across pricing, calibration, Greeks, and risk. See
  [`docs/technical_notes.md`](docs/technical_notes.md) for the full statement.
- **Reproducibility:** Monte Carlo work fixes a random seed by default (overridable for convergence
  studies), and every deliverable above is regenerated by running one script — no result exists only
  because it was once printed in a notebook cell.
- **No silent fallbacks:** if `yfinance` returns no data or a stale/incomplete chain, the pipeline
  fails loudly or falls back to the last successfully cached pull with a printed warning — it never
  fabricates market data.
- **Generated, not committed:** `data/cache/` and `results/` are gitignored; everything in them is
  reproducible from a clean checkout given network access.

## Technology

| Area | Tools |
| --- | --- |
| Language | Python ≥3.11, managed with [uv](https://docs.astral.sh/uv/) |
| Numerical computing | NumPy, SciPy |
| Market data | `yfinance` |
| Visualization | Matplotlib and Plotly |
| Interactive app | Streamlit |
| Testing | pytest |
| Cross-validation | QuantLib, for checking results only (dev dependency, never imported in `src/`) |

SciPy provides the optimization routines for SVI calibration (weighted least squares with nonlinear
no-arbitrage constraints, via SLSQP).

## Documentation requirements

The README and accompanying technical notes explicitly cover:

- Model assumptions: risk-neutral measure, log-normality, and no-arbitrage
- The limitations of Black-Scholes revealed by the observed volatility smile
- Why the control variate reduces variance, including a short mathematical proof
- The interpretation of theoretical versus real hedging costs

See [`docs/technical_notes.md`](docs/technical_notes.md) for all four, plus the SVI no-arbitrage checks.
