#!/usr/bin/env python3
"""
Example usage of the equity simulation library.

Demonstrates pricing European call and put options using Monte Carlo
simulation and compares with Black-Scholes analytical prices.
"""

import numpy as np

from lib import (
    GBMSimulator,
    GBMParameters,
    EuropeanCallPayoff,
    EuropeanPutPayoff,
    MonteCarloEngine,
)
from lib.payoffs import CustomPayoff, PayoffType
from lib.pricing import black_scholes_call, black_scholes_put


def main():
    # Market parameters
    s0 = 100.0  # Initial stock price
    k = 105.0  # Strike price
    t = 1.0  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)

    print("=" * 60)
    print("Equity Derivatives Monte Carlo Pricing")
    print("=" * 60)
    print(f"\nMarket Parameters:")
    print(f"  Initial price (S0): ${s0:.2f}")
    print(f"  Strike price (K):   ${k:.2f}")
    print(f"  Time to maturity:   {t:.2f} years")
    print(f"  Risk-free rate:     {r:.1%}")
    print(f"  Volatility:         {sigma:.1%}")

    # Create GBM simulator
    params = GBMParameters(s0=s0, mu=0.1, sigma=sigma, r=r)
    simulator = GBMSimulator(params, seed=42)

    # Create pricing engine
    engine = MonteCarloEngine(simulator, n_paths=100_000)

    # Price European Call
    print("\n" + "-" * 60)
    print("European Call Option")
    print("-" * 60)

    call_payoff = EuropeanCallPayoff(strike=k)
    call_result = engine.price(call_payoff, t)
    bs_call = black_scholes_call(s0, k, t, r, sigma)

    print(f"  Monte Carlo: {call_result}")
    print(f"  Black-Scholes: {bs_call:.6f}")
    print(f"  Difference: {abs(call_result.price - bs_call):.6f}")

    # Price with antithetic variates
    call_result_av = engine.price_with_control_variate(call_payoff, t)
    print(f"\n  With Antithetic Variates: {call_result_av}")
    print(f"  Variance reduction: {call_result.std_error / call_result_av.std_error:.2f}x")

    # Price European Put
    print("\n" + "-" * 60)
    print("European Put Option")
    print("-" * 60)

    put_payoff = EuropeanPutPayoff(strike=k)
    put_result = engine.price(put_payoff, t)
    bs_put = black_scholes_put(s0, k, t, r, sigma)

    print(f"  Monte Carlo: {put_result}")
    print(f"  Black-Scholes: {bs_put:.6f}")
    print(f"  Difference: {abs(put_result.price - bs_put):.6f}")

    # Verify put-call parity: C - P = S0 - K*exp(-rT)
    print("\n" + "-" * 60)
    print("Put-Call Parity Check")
    print("-" * 60)
    parity_lhs = call_result.price - put_result.price
    parity_rhs = s0 - k * np.exp(-r * t)
    print(f"  C - P = {parity_lhs:.6f}")
    print(f"  S0 - K*exp(-rT) = {parity_rhs:.6f}")
    print(f"  Difference: {abs(parity_lhs - parity_rhs):.6f}")

    # Demonstrate custom exotic payoff: Digital Call
    print("\n" + "-" * 60)
    print("Custom Payoff Example: Digital Call")
    print("-" * 60)

    digital_call = CustomPayoff(
        payoff_func=lambda paths, tg: (
            (paths if paths.ndim == 1 else paths[:, -1]) > k
        ).astype(float),
        payoff_type=PayoffType.TERMINAL,
    )

    digital_result = engine.price(digital_call, t)
    print(f"  Digital Call (pays $1 if S(T) > K): {digital_result}")

    # Demonstrate path simulation
    print("\n" + "-" * 60)
    print("Path Simulation Example")
    print("-" * 60)

    # Simulate a few paths
    small_sim = GBMSimulator(params, seed=123)
    paths = small_sim.simulate_paths(t=1.0, n_steps=12, n_paths=5)
    time_grid = small_sim.get_time_grid(t=1.0, n_steps=12)

    print(f"  Simulated {paths.shape[0]} paths with {paths.shape[1]} time points")
    print(f"  Time grid (monthly): {time_grid.round(2)}")
    print(f"  Sample path (first): {paths[0].round(2)}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
