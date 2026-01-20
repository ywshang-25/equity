"""
Monte Carlo pricing engine for equity derivatives.

Prices derivatives by computing the expected discounted payoff
under the risk-neutral measure using simulated stock price paths.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gbm import GBMSimulator, GBMParameters
from .payoffs import Payoff, PayoffType


@dataclass
class PricingResult:
    """Result of Monte Carlo pricing."""

    price: float  # Estimated option price
    std_error: float  # Standard error of the estimate
    n_paths: int  # Number of simulation paths used
    confidence_interval_95: tuple[float, float]  # 95% confidence interval

    def __str__(self) -> str:
        return (
            f"Price: {self.price:.6f} "
            f"(SE: {self.std_error:.6f}, "
            f"95% CI: [{self.confidence_interval_95[0]:.6f}, "
            f"{self.confidence_interval_95[1]:.6f}])"
        )


class MonteCarloEngine:
    """
    Monte Carlo engine for pricing equity derivatives.

    Prices derivatives by:
    1. Simulating stock price paths under the risk-neutral measure
    2. Evaluating the payoff for each path
    3. Discounting and averaging the payoffs

    The price estimate is: E[e^(-rT) * payoff(S)]
    """

    def __init__(
        self,
        simulator: GBMSimulator,
        n_paths: int = 100_000,
        n_steps: int = 252,
    ):
        """
        Initialize the Monte Carlo pricing engine.

        Args:
            simulator: GBM simulator for generating stock price paths
            n_paths: Number of Monte Carlo paths (default: 100,000)
            n_steps: Number of time steps for path-dependent options (default: 252)
        """
        self.simulator = simulator
        self.n_paths = n_paths
        self.n_steps = n_steps

    def price(
        self,
        payoff: Payoff,
        t: float,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> PricingResult:
        """
        Price a derivative using Monte Carlo simulation.

        Args:
            payoff: Payoff object defining the derivative
            t: Time to maturity (in years)
            n_paths: Number of paths (overrides default)
            n_steps: Number of time steps (overrides default, for path-dependent)

        Returns:
            PricingResult containing price, standard error, and confidence interval
        """
        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps
        r = self.simulator.params.r

        # Simulate paths based on payoff type
        if payoff.payoff_type == PayoffType.TERMINAL:
            # For terminal payoffs, only need S(T)
            paths = self.simulator.simulate_terminal(t, n_paths, risk_neutral=True)
            time_grid = None
        else:
            # For path-dependent payoffs, need full paths
            paths = self.simulator.simulate_paths(
                t, n_steps, n_paths, risk_neutral=True
            )
            time_grid = self.simulator.get_time_grid(t, n_steps)

        # Evaluate payoffs
        payoffs = payoff.evaluate(paths, time_grid)

        # Discount payoffs
        discount_factor = np.exp(-r * t)
        discounted_payoffs = discount_factor * payoffs

        # Compute statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)

        # 95% confidence interval (1.96 standard errors)
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return PricingResult(
            price=float(price),
            std_error=float(std_error),
            n_paths=n_paths,
            confidence_interval_95=(float(ci_lower), float(ci_upper)),
        )

    def price_with_control_variate(
        self,
        payoff: Payoff,
        t: float,
        n_paths: Optional[int] = None,
    ) -> PricingResult:
        """
        Price a derivative using Monte Carlo with antithetic variates.

        Uses antithetic sampling to reduce variance: for each Z, also use -Z.

        Args:
            payoff: Payoff object defining the derivative
            t: Time to maturity (in years)
            n_paths: Number of paths (overrides default, will be doubled)

        Returns:
            PricingResult with reduced variance estimate
        """
        n_paths = n_paths or self.n_paths
        half_paths = n_paths // 2
        r = self.simulator.params.r
        sigma = self.simulator.params.sigma
        s0 = self.simulator.params.s0

        # Generate random variables
        z = self.simulator.rng.standard_normal(half_paths)

        # Simulate with Z and -Z (antithetic)
        drift_term = (r - 0.5 * sigma**2) * t
        vol_term = sigma * np.sqrt(t)

        st_pos = s0 * np.exp(drift_term + vol_term * z)
        st_neg = s0 * np.exp(drift_term + vol_term * (-z))

        # Combine paths
        all_paths = np.concatenate([st_pos, st_neg])

        # Evaluate payoffs
        payoffs = payoff.evaluate(all_paths, None)

        # Discount payoffs
        discount_factor = np.exp(-r * t)
        discounted_payoffs = discount_factor * payoffs

        # Antithetic estimator: average of paired samples
        payoffs_pos = discounted_payoffs[:half_paths]
        payoffs_neg = discounted_payoffs[half_paths:]
        antithetic_payoffs = 0.5 * (payoffs_pos + payoffs_neg)

        # Compute statistics
        price = np.mean(antithetic_payoffs)
        std_error = np.std(antithetic_payoffs, ddof=1) / np.sqrt(half_paths)

        # 95% confidence interval
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return PricingResult(
            price=float(price),
            std_error=float(std_error),
            n_paths=n_paths,
            confidence_interval_95=(float(ci_lower), float(ci_upper)),
        )


def black_scholes_call(s0: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Analytical Black-Scholes price for European call option.

    Useful for validating Monte Carlo results.
    """
    from scipy.stats import norm

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    return s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def black_scholes_put(s0: float, k: float, t: float, r: float, sigma: float) -> float:
    """
    Analytical Black-Scholes price for European put option.

    Useful for validating Monte Carlo results.
    """
    from scipy.stats import norm

    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    return k * np.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
