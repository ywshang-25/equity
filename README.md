# Equity Derivatives Monte Carlo Library

A Monte Carlo simulation framework for pricing equity derivatives using Geometric Brownian Motion (GBM) for stock price dynamics.

> Created by Yanwen Shang's instructions with implementation by Claude Code.
> This public repository illustrates key components and concepts using public-domain knowledge only. Advanced features may be available in private repositories.

## Overview

The library prices European and American options (and arbitrary exotic payoffs) by simulating stock price paths under the risk-neutral measure and computing expected discounted payoffs. American options use the **Longstaff-Schwartz (LSM)** algorithm for early-exercise decisions, with Cox-Ross-Rubinstein binomial trees available as a benchmark.

## Structure

```
lib/
  gbm.py       # GBM simulator — exact-solution path generation
  payoffs.py   # Payoff classes (European, American, custom)
  pricing.py   # MonteCarloEngine, Black-Scholes, binomial tree formulas
tests/
  test_gbm.py
  test_payoffs.py
  test_pricing.py
example.py     # Runnable demo
```

## Payoffs

| Class | Type | Payoff |
|---|---|---|
| `EuropeanCallPayoff(strike)` | Terminal | max(S(T) − K, 0) |
| `EuropeanPutPayoff(strike)` | Terminal | max(K − S(T), 0) |
| `AmericanCallPayoff(strike)` | Path-dependent | max(S(t) − K, 0), early exercise |
| `AmericanPutPayoff(strike)` | Path-dependent | max(K − S(t), 0), early exercise |
| `CustomPayoff(func)` | Either | User-defined function |

## Usage

### European options

```python
from lib import GBMParameters, GBMSimulator, MonteCarloEngine, EuropeanCallPayoff, EuropeanPutPayoff
from lib.pricing import black_scholes_call

params = GBMParameters(s0=100, mu=0.1, sigma=0.2, r=0.05)
simulator = GBMSimulator(params, seed=42)
engine = MonteCarloEngine(simulator, n_paths=100_000)

call = EuropeanCallPayoff(strike=105)
result = engine.price(call, t=1.0)
print(result)               # Price, SE, 95% CI
print(black_scholes_call(100, 105, 1.0, 0.05, 0.2))  # Analytical benchmark
```

### American options (Longstaff-Schwartz)

```python
from lib import AmericanCallPayoff, AmericanPutPayoff
from lib.pricing import binomial_tree_american_put

am_put = AmericanPutPayoff(strike=105)
result = engine.price_american(am_put, t=1.0)
print(result)                                          # LSM price
print(binomial_tree_american_put(100, 105, 1.0, 0.05, 0.2))  # Binomial benchmark
```

### Variance reduction

```python
# Antithetic variates — typically halves the standard error
result = engine.price_with_control_variate(call, t=1.0)
```

### Custom / exotic payoffs

```python
from lib.payoffs import CustomPayoff, PayoffType
import numpy as np

# Asian call (arithmetic average)
asian_call = CustomPayoff(
    payoff_func=lambda paths, tg: np.maximum(np.mean(paths, axis=1) - 105, 0),
    payoff_type=PayoffType.PATH_DEPENDENT,
)
result = engine.price(asian_call, t=1.0)
```

## Pricing methods

| Method | Applies to | Benchmark |
|---|---|---|
| `engine.price(payoff, t)` | European / exotic | Black-Scholes (`black_scholes_call/put`) |
| `engine.price_with_control_variate(payoff, t)` | European (terminal) | — |
| `engine.price_american(payoff, t)` | American | `binomial_tree_american_call/put` |

## Dependencies

- Python 3.10+
- NumPy >= 2.0
- SciPy >= 1.10
- pytest >= 7.0 (tests only)

Install with:

```bash
pip install -r requirements.txt
```

## Running the example

```bash
python example.py
```

## Running the tests

```bash
python -m pytest tests/ -v
```
