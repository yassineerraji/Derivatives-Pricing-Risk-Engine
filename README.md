# Derivatives Pricing & Risk Engine

An options pricing and risk management engine that implements several numerical methods, calibrates to real market data, and measures the practical cost of hedging.

> **Project brief**
> Build the models from first principles, compare their accuracy and performance, and use market data to connect theory with a realistic risk workflow.

## Contents

- [Scope](#scope)
- [Modules](#modules)
- [Technology](#technology)
- [Documentation requirements](#documentation-requirements)

## Scope

The engine will cover:

- European option pricing with three independent numerical approaches
- Implied volatility extraction and SVI volatility-surface calibration
- Greeks computed analytically, by finite differences, and by Monte Carlo
- Historical and model-based VaR, Expected Shortfall, and discrete delta hedging

The Black-Scholes implementation is the baseline for validation, not a substitute for the numerical methods being developed.

## Modules

### 1. Pricing

Compare three pricing methods using price, runtime, and error against the closed-form Black-Scholes result.

| Method | Implementation | Purpose |
| --- | --- | --- |
| **Black-Scholes** | Closed-form European call and put | Validation reference |
| **Monte Carlo** | Risk-neutral GBM with antithetic and control variates | Quantify convergence and variance reduction |
| **Finite differences** | Crank-Nicolson scheme for the Black-Scholes PDE | Compare PDE accuracy and runtime with Monte Carlo |

The Monte Carlo analysis should include confidence intervals, variance before and after reduction, and a convergence plot of error against the number of simulations. The finite-difference comparison should use a common tolerance level.

**Deliverable:** A comparative table covering price, computation time, and error versus closed-form Black-Scholes.

### 2. Calibration to real data

1. Pull an options chain from Yahoo Finance (`yfinance`) or CBOE when accessible, using a liquid underlying such as SPY or AAPL.
2. Extract implied volatility for each strike and maturity with Newton-Raphson or Brent root-finding.
3. Calibrate an SVI (Stochastic Volatility Inspired) model to parameterize the volatility smile.
4. Visualize the resulting three-dimensional volatility surface across strike, maturity, and implied volatility.

**Deliverable:** A calibrated volatility surface and calibration code, including the weighted-least-squares cost function.

### 3. Greeks

Compare the following approaches for delta, gamma, vega, theta, and rho where applicable:

- **Analytical:** Closed-form Black-Scholes formulas
- **Finite differences:** Bump-and-reprice with a documented step size $h$ and a discussion of bias versus variance
- **Monte Carlo:** Pathwise and likelihood-ratio estimators, with delta and vega implemented at minimum

**Deliverable:** A comparative table of accuracy and computation cost for each Greek and method.

### 4. Risk management

Build a simplified book containing 5-10 option positions across one or two underlyings.

- **VaR and Expected Shortfall:** Compute both historical-simulation estimates from real returns and Monte Carlo estimates from the calibrated model.
- **Discrete delta hedging:** Simulate daily rebalancing with bid-ask spread and linear market-impact costs.
- **P&L attribution:** Compare frictionless Black-Scholes replication P&L with P&L including transaction costs to quantify the real cost of hedging.

**Deliverable:** A cumulative hedging P&L chart comparing theoretical and frictional results, plus VaR and Expected Shortfall with confidence intervals.

## Technology

| Area | Tools |
| --- | --- |
| Language | Python |
| Numerical computing | NumPy, SciPy |
| Market data | `yfinance` |
| Visualization | Matplotlib and/or Plotly |
| Cross-validation | QuantLib, for checking results only |

SciPy will provide the optimization routines for SVI calibration. QuantLib must remain a cross-validation tool: the pricing, calibration, and risk models should be implemented directly in this project.

## Documentation requirements

The README and accompanying technical notes should explicitly cover:

- Model assumptions: risk-neutral measure, log-normality, and no-arbitrage
- The limitations of Black-Scholes revealed by the observed volatility smile
- Why the control variate reduces variance, including a short mathematical proof
- The interpretation of theoretical versus real hedging costs
