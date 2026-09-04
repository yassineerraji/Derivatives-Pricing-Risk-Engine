# CLAUDE.md

Implementation guide for building the **Derivatives Pricing & Risk Engine** described in [README.md](README.md). Read the README first — it is the spec. This file is the build plan and engineering conventions on top of it.

## Non-negotiables

- **Pricing, calibration, Greeks, and risk models are implemented from scratch** (NumPy/SciPy only). QuantLib is a cross-validation dependency — it may appear in `tests/` to assert your results against it, never inside `src/`.
- Black-Scholes closed-form is the **validation baseline** for every other method, not a component to skip past.
- Every deliverable in the README (comparison tables, plots, calibrated surface, P&L chart) must be reproducible by running one script — no results that only exist because they were once printed in a notebook cell.
- Fix random seeds for all Monte Carlo work by default (seed as a parameter, default fixed), so runs are reproducible; allow overriding for convergence studies.

## Environment

- Python ≥3.11, managed with **uv**. `pyproject.toml` + `uv.lock` at the repo root.
- Core deps: `numpy`, `scipy`, `yfinance`, `matplotlib` (or `plotly`), `pandas`.
- Dev deps: `pytest`, `quantlib-python` (cross-validation only).
- Run everything via `uv run ...`; add scripts as `[project.scripts]` entries once stable.

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
  greeks/
    analytical.py           # closed-form Greeks
    finite_difference.py    # bump-and-reprice
    monte_carlo.py           # pathwise + likelihood-ratio estimators
  risk/
    book.py                  # option book construction (5-10 positions)
    var_es.py                # historical-simulation and MC VaR/ES
    hedging.py                # discrete delta-hedge simulation with costs
    pnl.py                    # frictionless vs frictional P&L attribution

scripts/                    # one entrypoint per README deliverable, thin orchestration only
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
  technical_notes.md         # the four documentation-requirements topics from the README
```

Scripts contain no pricing/math logic themselves — they call `src/dpre`, write to `results/`, and are what "the deliverable" means operationally.

## Build order

Work through phases in order; each has a concrete exit condition. Don't start a phase's deliverable script until its `src/` logic has passing tests.

### Phase 0 — Scaffolding
- `uv init`, `pyproject.toml`, package skeleton above, `pytest` wired up, `.gitignore` (data/cache, results, `.venv`).
- Exit: `uv run pytest` runs (even with zero tests) and package imports cleanly.

### Phase 1 — Pricing (README §1)
1. `black_scholes.py`: closed-form European call/put. Test: put-call parity holds to float tolerance across a grid of strikes/maturities.
2. `monte_carlo.py`: risk-neutral GBM terminal simulation; antithetic variates; a control variate (e.g. the underlying itself, whose expectation is known analytically). Track variance with/without reduction. Test: MC price converges to BS price as paths → ∞, within a stated confidence interval.
3. `finite_difference.py`: Crank-Nicolson for the BS PDE, one grid builder reusable for calls and puts. Test: FD price matches BS within a documented tolerance at a fixed grid resolution.
4. `scripts/01_pricing_comparison.py`: builds the comparative table (price, runtime, error vs BS) for all three methods, plus the MC convergence plot (error vs number of simulations) and the variance-reduction before/after comparison. Writes to `results/tables/pricing_comparison.csv` and `results/plots/mc_convergence.png`.
- Exit: script runs end-to-end and produces both artifacts; tests green.

### Phase 2 — Calibration to real data (README §2)
1. `calibration/data.py`: pull an options chain via `yfinance` for one liquid underlying (default SPY, configurable). Cache raw pulls under `data/cache/` keyed by ticker+date so repeated runs don't re-hit the API. Handle sparse/missing strikes (drop, don't fabricate).
2. `calibration/implied_vol.py`: IV per (strike, maturity) via Brent's method (robust) with Newton-Raphson as the documented alternative. Test: round-trip — price at a known vol, recover that vol.
3. `calibration/svi.py`: SVI smile parameterization, `scipy.optimize` weighted-least-squares fit per maturity slice, weights documented (e.g. by vega or bid-ask width). Test: fit converges and reprices the smile within tolerance on synthetic SVI-generated data.
4. `scripts/02_calibrate_surface.py`: fetches data, extracts IV surface, fits SVI per maturity, plots the 3D surface (strike × maturity × IV). Writes `results/tables/svi_params.csv` and `results/plots/vol_surface.png`.
- Exit: script produces a calibrated surface from live (or last-cached) market data.

### Phase 3 — Greeks (README §3)
1. `greeks/analytical.py`: closed-form delta, gamma, vega, theta, rho.
2. `greeks/finite_difference.py`: bump-and-reprice with a fixed, documented step `h` per Greek; note bias/variance tradeoff in a docstring or `docs/technical_notes.md`, not inline prose.
3. `greeks/monte_carlo.py`: pathwise estimator (delta, vega at minimum) and likelihood-ratio estimator; reuse `pricing/monte_carlo.py`'s path generator rather than duplicating it.
4. `scripts/03_greeks_comparison.py`: table of accuracy (vs analytical) and computation cost per Greek × method. Writes `results/tables/greeks_comparison.csv`.
- Exit: table generated; MC Greek estimators pass a tolerance test against analytical values.

### Phase 4 — Risk management (README §4)
1. `risk/book.py`: construct a 5-10 position book across 1-2 underlyings (mix of calls/puts, strikes, maturities) — define as data, not a UI.
2. `risk/var_es.py`: historical-simulation VaR/ES from real return history, and MC VaR/ES using the calibrated model from Phase 2. Both with confidence intervals.
3. `risk/hedging.py`: daily-rebalanced delta hedge simulation with bid-ask spread and linear market-impact cost, driven by real (or realistically simulated) price paths.
4. `risk/pnl.py`: frictionless BS-replication P&L vs frictional P&L; the difference is the deliverable.
5. `scripts/04_risk_report.py`: cumulative hedging P&L chart (theoretical vs frictional) plus VaR/ES table with CIs. Writes `results/plots/hedging_pnl.png`, `results/tables/var_es.csv`.
- Exit: report script runs against the Phase 4 book and Phase 2 calibration output.

### Phase 5 — Documentation
- `docs/technical_notes.md` explicitly covers, per the README's Documentation requirements: risk-neutral/log-normal/no-arbitrage assumptions; what the observed smile reveals about BS's limitations; the control-variate variance-reduction proof; the interpretation of theoretical vs real hedging cost. Link out from the README rather than duplicating.

## Conventions

- **Interfaces**: pricing/Greek functions take explicit `(S, K, T, r, sigma, ...)` args, not opaque config objects — this is a numerics project, keep signatures mathematically legible. Introduce a shared `MarketParams`/`OptionSpec` dataclass only once ≥3 functions pass the identical tuple around, not before.
- **Docstrings** (overrides the usual "no comments" default for this project): every file opens with a one-line module docstring stating its role; every function gets a short docstring (one line, or a couple if inputs/outputs need units/convention noted, e.g. `T` in years, `sigma` annualized). Keep them factual and brief — not full NumPy-style sections.
- **Day count / compounding**: ACT/365, continuously-compounded rates, annualized vol — state this once in `docs/technical_notes.md` and follow it everywhere; don't re-derive per module.
- **Vectorization**: NumPy-vectorize path generation and payoff evaluation; avoid Python-level loops over simulation paths.
- **Testing**: every numerical method gets a correctness test against a known closed form or a convergence/tolerance bound — not just "it runs." Cross-validate calibration and at least one pricing method against QuantLib in `tests/`, isolated from `src/`.
- **No silent fallbacks**: if `yfinance` returns no data or a stale/incomplete chain, fail loudly or use the last successfully cached pull with a printed warning — never fabricate market data.
- **Results are generated, not committed**: `data/cache/` and `results/` are gitignored; scripts must be able to regenerate everything from a clean checkout (network access permitting).
