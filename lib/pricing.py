"""
Monte Carlo pricing engine for equity derivatives.

Prices derivatives by computing the expected discounted payoff
under the risk-neutral measure using simulated stock price paths.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gbm import GBMSimulator, GBMParameters
from .payoffs import Payoff, PayoffType, AmericanPayoff


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


    def price_american(
        self,
        payoff: AmericanPayoff,
        t: float,
        n_paths: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> PricingResult:
        """
        Price an American option using the Longstaff-Schwartz (LSM) algorithm.

        LSM estimates the continuation value at each exercise date via
        least-squares regression of discounted future cash flows on a
        polynomial basis of the current stock price, then exercises early
        whenever the intrinsic value exceeds the estimated continuation value.

        Reference: Longstaff & Schwartz (2001), "Valuing American Options
        by Simulation: A Simple Least-Squares Approach", RFS 14(1).

        Args:
            payoff: AmericanCallPayoff or AmericanPutPayoff instance
            t: Time to maturity (in years)
            n_paths: Number of paths (overrides default)
            n_steps: Number of time steps (overrides default)

        Returns:
            PricingResult with price, standard error, and confidence interval
        """
        if not isinstance(payoff, AmericanPayoff):
            raise TypeError("price_american requires an AmericanPayoff instance")

        n_paths = n_paths or self.n_paths
        n_steps = n_steps or self.n_steps
        r = self.simulator.params.r
        dt = t / n_steps
        discount = np.exp(-r * dt)

        # Simulate full price paths under the risk-neutral measure
        paths = self.simulator.simulate_paths(t, n_steps, n_paths, risk_neutral=True)

        # Option value vector, initialised to the terminal (maturity) payoff
        V = payoff.intrinsic_value(paths[:, -1]).astype(float)

        # Backward induction: Longstaff-Schwartz
        for step in range(n_steps - 1, 0, -1):
            V *= discount  # Discount continuation value one period

            S = paths[:, step]
            intrinsic = payoff.intrinsic_value(S)
            itm = intrinsic > 0  # Only regress on in-the-money paths

            if itm.sum() >= 3:  # Need enough points for a stable regression
                S_itm = S[itm]
                Y = V[itm]  # Discounted continuation values for ITM paths

                # Polynomial basis: 1, S, S^2
                X = np.column_stack([np.ones_like(S_itm), S_itm, S_itm**2])
                coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                estimated_continuation = X @ coeffs

                # Exercise where intrinsic exceeds estimated continuation
                exercise = intrinsic[itm] > estimated_continuation
                V[np.where(itm)[0][exercise]] = intrinsic[itm][exercise]

        # One final discount from step 1 to time 0
        V *= discount

        price = float(np.mean(V))
        std_error = float(np.std(V, ddof=1) / np.sqrt(n_paths))
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        return PricingResult(
            price=price,
            std_error=std_error,
            n_paths=n_paths,
            confidence_interval_95=(ci_lower, ci_upper),
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


def binomial_tree_american_call(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    n_steps: int = 500,
) -> float:
    """
    Cox-Ross-Rubinstein binomial tree price for an American call option.

    Useful for validating Longstaff-Schwartz results. Note: for a
    non-dividend-paying stock this equals the European (Black-Scholes) call.

    Args:
        s0: Initial stock price
        k: Strike price
        t: Time to maturity (in years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        n_steps: Number of binomial steps (default: 500)

    Returns:
        American call option price
    """
    dt = t / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal stock prices and payoffs
    j = np.arange(n_steps + 1)
    S_T = s0 * u ** (n_steps - j) * d**j
    V = np.maximum(S_T - k, 0.0)

    # Backward induction with early-exercise check
    for i in range(n_steps - 1, -1, -1):
        j_i = np.arange(i + 1)
        S_i = s0 * u ** (i - j_i) * d**j_i
        V = disc * (p * V[: i + 1] + (1 - p) * V[1 : i + 2])
        V = np.maximum(V, S_i - k)

    return float(V[0])


def binomial_tree_american_put(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    n_steps: int = 500,
) -> float:
    """
    Cox-Ross-Rubinstein binomial tree price for an American put option.

    Useful for validating Longstaff-Schwartz results.

    Args:
        s0: Initial stock price
        k: Strike price
        t: Time to maturity (in years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        n_steps: Number of binomial steps (default: 500)

    Returns:
        American put option price
    """
    dt = t / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    S_T = s0 * u ** (n_steps - j) * d**j
    V = np.maximum(k - S_T, 0.0)

    for i in range(n_steps - 1, -1, -1):
        j_i = np.arange(i + 1)
        S_i = s0 * u ** (i - j_i) * d**j_i
        V = disc * (p * V[: i + 1] + (1 - p) * V[1 : i + 2])
        V = np.maximum(V, k - S_i)

    return float(V[0])
