# Technical notes

Companion to the [README](../README.md); covers the four documentation requirements it lists.
See [CLAUDE.md](../CLAUDE.md) for the module-by-module build plan these notes reference.

## 1. Model assumptions: risk-neutral measure, log-normality, no-arbitrage

Every pricing, calibration, and Greeks module (`src/dpre/pricing/`, `src/dpre/greeks/`,
`src/dpre/calibration/`) prices under the same three linked assumptions:

- **No-arbitrage.** No self-financing trading strategy earns a riskless profit at zero cost. This
  is the assumption that *lets* a fair price exist at all, and is what put-call parity tests
  (`tests/pricing/test_black_scholes.py`) and the SVI arbitrage checks (§3 below) actually verify.
- **Risk-neutral valuation.** Absence of arbitrage implies an equivalent martingale measure `Q`
  under which every discounted asset price is a martingale; a claim's price is the `Q`-expectation
  of its discounted payoff. This is why every pricer here drifts the underlying at `r - q` (the
  risk-free rate net of dividend yield), never at a real-world expected return `mu` — `mu` doesn't
  appear anywhere in `black_scholes.py`, `monte_carlo.py`, or `finite_difference.py`.
- **Log-normality.** Black-Scholes-Merton further assumes `dS = (r-q)S dt + sigma S dW` under `Q`,
  so `S_T` is log-normal with constant `sigma`. This is the assumption §2 shows the market
  violates — which is exactly why Phase 2 calibrates a smile instead of using one flat `sigma`.

**Where this project deliberately steps outside `Q`:** `risk/var_es.py`'s historical VaR/ES replays
*real* historical returns (the physical measure `P`) against the book, because VaR asks "how much
could I actually lose," not "what is the fair hedging price" — those are different questions and
use different measures on purpose. Its Monte Carlo counterpart stays closer to `Q`: it simulates
returns at the calibrated (risk-neutral-implied) near-term ATM vol as a proxy for near-term
realized vol, an approximation worth naming rather than leaving implicit — implied vol embeds a
variance risk premium and is not, in general, an unbiased forecast of realized vol.

## 2. What the observed smile reveals about Black-Scholes' limitations

If Black-Scholes' constant-`sigma` assumption held, every strike and maturity on a given underlying
would imply the *same* vol when inverted through the BS formula. Phase 2's calibration
(`results/plots/vol_surface.png`) shows the opposite: implied vol varies substantially by strike
*and* maturity for SPY — a direct, visual falsification of the model's core assumption, not a
subtle statistical effect.

Two specific shapes are visible and both have standard economic explanations BS cannot produce on
its own (constant `sigma` has no mechanism for either):

- **Negative skew** (OTM puts price at higher implied vol than OTM calls of equal moneyness): the
  leverage effect (a falling stock raises a firm's leverage, raising the volatility of its equity)
  and crash/jump risk being priced into downside insurance (OTM puts) more than the model's
  continuous-diffusion assumption would predict.
- **Term structure** (implied vol differs by maturity at fixed moneyness): incompatible with a
  single constant `sigma` by definition; reflects the market pricing in mean-reverting or
  regime-dependent volatility that GBM cannot represent.

**A distinction worth being precise about:** the smile's existence is a real, economically
meaningful market fact — it is a separate question from whether a *fit to it* happens to violate
static no-arbitrage conditions (§3 covers that; a plain unconstrained per-slice fit did violate
them here initially, which was an artifact of the fitting procedure on noisy quotes, not evidence
that the real market is itself arbitrageable). Conflating "the smile is real" with "a particular fit
to it has arbitrage" would be wrong in both directions.

**Practical consequence for this project:** since a single flat `sigma` misprices anything away
from the strike/maturity it was fit to, `risk/valuation.py` prices every book position off the
*smile* (nearest-maturity SVI slice, sticky-strike) rather than one global `sigma` — the whole
reason Phase 2's calibration feeds Phase 4's risk numbers instead of Phase 4 inventing its own vol.
Black-Scholes still earns its role as the baseline, though: it is arbitrage-consistent by
construction, and "implied vol" is itself *defined* by inverting the BS formula against a market
price — the model remains the market's shared quoting convention even though its own generative
assumption (constant vol) is known to be false.

## 3. SVI calibration: no-arbitrage checks

`src/dpre/calibration/arbitrage.py` checks the calibrated surface for the two classic forms of
static arbitrage:

- **Butterfly arbitrage** (per slice): Durrleman's condition, `g(k) >= 0` for all log-moneyness
  `k`, where `g` is built from the total-variance function and its first two derivatives (see the
  module docstring). `g(k) < 0` somewhere means the smile implies a negative risk-neutral density
  at some strike via Breeden-Litzenberger — a static arbitrage.
- **Calendar arbitrage** (across consecutive maturities): total variance at a fixed strike must be
  non-decreasing in `T`. A violation means a calendar spread (short the earlier expiry, long the
  later one, same strike) would have a guaranteed non-negative payoff at zero cost.

Both are unit-tested against hand-built slices with a known-good and a deliberately pathological
parameterization (`tests/calibration/test_arbitrage.py`).

**Original finding (independent per-slice least squares):** run against a live SPY chain (8
maturities, ~4 days to ~10 months — `_select_expiries` caps the calibrated horizon at one year), 5
of 8 slices failed the butterfly check and 4 of 7 consecutive
maturity pairs failed the calendar check. That was expected, not a bug: fitting each maturity
slice **independently** by unconstrained weighted least squares gives the objective nothing that
penalizes a locally non-convex or maturity-decreasing total-variance curve — it only minimizes fit
error against noisy, partly `last_price`-derived quotes (see
[`data.py`](../src/dpre/calibration/data.py)).

**Fix: constrained sequential calibration.** `calibrate_svi_surface` now fits slices in increasing
`T` order via `calibrate_svi_slice_arbitrage_free` (SLSQP), which adds two nonlinear constraints to
the same weighted-least-squares objective, each evaluated on a fixed log-moneyness grid
(`DEFAULT_ARBITRAGE_CHECK_GRID`, ±1.5 in `k`) with a small positive margin (`ARBITRAGE_MARGIN`) so
that solver tolerance and grid-vs-checker-grid mismatch can't leave a fit "just barely" infeasible
once re-verified by `arbitrage.py`'s own (denser, independently ranged) grid:

- `durrleman_g(k, ...) >= margin` on the grid (no butterfly arbitrage in this slice alone).
- (from the second slice on) `w(k) - w_prev(k) >= margin` on the grid, against the *already-fitted*
  previous maturity (no calendar arbitrage against it).

This trades a small amount of fit quality for a surface that's actually usable for risk without the
caveat above — `tests/calibration/test_svi.py` includes a case with deliberately calendar-violating
*input* data (a longer maturity's target variance set below a shorter one's) and confirms the
constrained fit still recovers an arbitrage-free pair of slices from it. Re-run against the same
live SPY chain: **all 8 slices now pass both checks**
(`results/tables/svi_arbitrage_check.csv`), and the fitted curves still visibly track the market
quotes in `results/plots/vol_surface.png`.

This is still a per-slice-sequential, not a joint/surface-level (e.g. SSVI), approach — each slice
individually satisfies both conditions against its immediate predecessor, which is sufficient for
the pairwise calendar check `arbitrage.py` runs, but a true joint parameterization would guarantee
this by construction rather than by per-slice constrained optimization. Noted here as the remaining
gap between "passes the checks we run" and "provably arbitrage-free by construction."

## 4. Why the control variate reduces variance, with proof

`pricing/monte_carlo.py` estimates `theta = E[Y]` (the discounted payoff's expectation) using
`X = S_T` as a control variate, whose mean is known in closed form under `Q`:
`E[S_T] = S0 * exp((r-q)*T)`. The adjusted estimator, for any constant `b`, is

```
Y_c(b) = Y - b * (X - E[X])
```

**Unbiasedness, for any `b`:** `E[Y_c(b)] = E[Y] - b*(E[X] - E[X]) = E[Y] = theta`. The control
variate can never bias the estimate — `b` only affects its variance.

**Variance, as a function of `b`:**

```
Var(Y_c(b)) = Var(Y) - 2*b*Cov(X,Y) + b^2*Var(X)
```

a upward-opening parabola in `b`, minimized where its derivative vanishes:
`-2*Cov(X,Y) + 2*b*Var(X) = 0`, giving the optimal coefficient

```
b* = Cov(X,Y) / Var(X)
```

Substituting back:

```
Var(Y_c(b*)) = Var(Y) - Cov(X,Y)^2/Var(X) = Var(Y) * (1 - rho^2)
```

where `rho = Corr(X,Y)`. Since `0 <= rho^2 <= 1`, `Var(Y_c(b*)) <= Var(Y)` **always**, with equality
only when `X` and `Y` are uncorrelated (`rho=0` — a useless control variate) and the reduction
approaching total as `|rho| -> 1`.

**Why `S_T` is a good control variate here:** for a call, the payoff `max(S_T-K, 0)` is a
non-decreasing function of `S_T`, so the two are strongly positively correlated (though not
perfectly — the payoff's kink at `K` keeps `rho` below 1). `mc_price` in `monte_carlo.py` estimates
`b*` by its sample analogue (`np.cov(payoff, S_T) / np.var(S_T)`) rather than the true `b*`, which
is a standard plug-in that introduces a negligible finite-sample bias (both estimate and control
draw from the same sample) but doesn't affect the asymptotic result above. Phase 1's measured
variance backs this up directly (`results/tables/mc_variance_reduction.csv`, ATM SPY call,
200k paths): raw variance 189.1, control-variate-only 34.2 — a ~5.5x reduction, consistent with a
moderately-high `rho` between the call payoff and `S_T` (antithetic variates give a further,
independent reduction on top, down to 33.3 combined).

## 5. Interpreting theoretical vs. real hedging cost

The classical Black-Scholes replication argument (Itô's lemma applied to a continuously
delta-hedged portfolio, the same derivation that produces the BS PDE) says that if you hedge
*continuously* at the *same* vol used to price the option, the replicating portfolio's P&L is
exactly zero — the option premium is precisely the cost of manufacturing its payoff synthetically.
`risk/hedging.py` and `risk/pnl.py` depart from that idealization, and — importantly — `pnl.py`
runs the comparison across `n_paths` independent simulated paths, not one, because a single path's
cumulative P&L is noisy enough to be actively misleading on its own (an earlier single-path version
of this simulation happened to land near a flat +$306; the properly-averaged 200-path result below
shows that draw was not representative of the typical outcome at all).

**Finding (200 paths, 50 trading days, live SPY book/surface):** the *frictionless* run's final P&L
has **mean +$5,200, std $4,094** across paths — nowhere near flat, and with a spread wide enough
that individual paths swing from solidly negative to over +$10,000. Three distinct effects are
mixed together in that number, and disentangling them is the point of this section:

1. **Smile/vol-mismatch P&L (the dominant effect here) — not a bug, not noise, but not
   "frictionless cost" either.** `risk/valuation.py` prices each book position off *its own*
   SVI-implied vol (sticky-strike), and this book's eight legs span 14.1%-21.2% vol. A single
   simulated price path, however, has exactly one realized vol. `pnl.py` uses the book's
   vega-weighted average implied vol (16.0%) to simulate that path — the standard choice for
   picking one representative number — but no single choice can make a *dispersion* of priced vols
   agree with a single realized-vol path. A leg priced (and hedged) at 21% vol, realizing a path
   with a lower actual vol, earns a systematic gain if we're net short that leg's gamma (or a loss
   if net long) — exactly the "vega/smile P&L" real trading desks track separately from cost. I
   verified this isn't an accounting bug two ways before accepting the number: tracing a single
   path's day-by-day P&L shows smooth accumulation with no discontinuous jumps (ruling out, e.g., a
   position's assigned SVI slice silently changing mid-run — it doesn't, `vol_for` keys off each
   position's fixed original `T`, never its shrinking `T_remaining`), and a controlled single-option
   book hedged at *exactly* its own pricing vol shows only a ~0.3%-of-notional mean bias — consistent
   with effect (2) below and nothing larger.
2. **Discretization ("gamma") noise.** Rebalancing once a day, not continuously, leaves the hedge
   imperfectly delta-neutral between rebalances. This is genuinely small on its own (the controlled
   single-option, matched-vol test above), but it's what the path-to-path *spread* around the mean
   (the ± $4,094 std, and the shaded band in `results/plots/hedging_pnl.png`) is mostly made of once
   effect (1)'s mean is accounted for.
3. **Transaction cost — the one this section is actually trying to isolate.** `HedgeCostModel`
   charges half the bid-ask spread plus a market-impact term on every rebalance. Both the
   frictionless and frictional runs trade *identical* share quantities along the *same* simulated
   path in each iteration (verified exactly, accounting for cash's risk-free compounding, in
   `tests/risk/test_hedging.py::test_frictional_cash_lags_frictionless_by_compounded_cost`) — cost
   is the only thing that differs between the two runs, which is what makes it possible to isolate
   despite effects (1) and (2) both being present in absolute terms. In the live run: frictional
   final P&L is **mean +$4,165, std $4,013** — systematically below frictionless by a mean gap of
   about $1,035, matching mean cumulative transaction cost of **$1,053 (std $207)** to within the
   compounding adjustment `test_hedging.py` derives.

**Interpretation:** effects (1) and (2) are real properties of hedging a multi-strike, multi-maturity
book with plain delta-only hedging against realized markets that don't share one flat vol — a desk
sees them too, and manages them with vega/vanna hedging this project doesn't implement. Effect (3)
is the specific "real cost of hedging" the README asks for, and it survives cleanly underneath the
other two: whatever a book's smile-driven P&L happens to be on a given realization, transaction
costs make it systematically worse by a predictable amount. The BS/SVI price is a *frictionless
replication cost* which desks must mark up over to cover *both* the transaction costs demonstrated
here *and* the smile-mismatch risk of not being able to vega-hedge every leg at its own vol — one
reason bid-ask spreads widen for the options that are most expensive to hedge (high-gamma,
near-the-money, near-expiry). The market-impact term's quadratic-in-size cost also means hedging a
large book costs disproportionately more than hedging several smaller ones separately — a real
effect (price impact), even though this project's linear-impact functional form is a standard
simplification, not an empirically fit one.
