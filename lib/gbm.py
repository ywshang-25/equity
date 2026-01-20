"""
Geometric Brownian Motion (GBM) simulation engine for stock prices.

The GBM model assumes stock prices follow:
    dS = μS dt + σS dW

where:
    S = stock price
    μ = drift (expected return)
    σ = volatility
    dW = Wiener process increment
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GBMParameters:
    """Parameters for Geometric Brownian Motion simulation."""

    s0: float  # Initial stock price
    mu: float  # Drift (annualized expected return)
    sigma: float  # Volatility (annualized)
    r: float = 0.0  # Risk-free rate (for risk-neutral simulation)

    def __post_init__(self):
        if self.s0 <= 0:
            raise ValueError("Initial stock price must be positive")
        if self.sigma < 0:
            raise ValueError("Volatility cannot be negative")


class GBMSimulator:
    """
    Simulator for generating stock price paths using Geometric Brownian Motion.

    Uses the exact solution for GBM:
        S(t) = S(0) * exp((μ - σ²/2)t + σW(t))

    For risk-neutral pricing, use mu = r (risk-free rate).
    """

    def __init__(self, params: GBMParameters, seed: Optional[int] = None):
        """
        Initialize the GBM simulator.

        Args:
            params: GBM parameters (s0, mu, sigma, r)
            seed: Random seed for reproducibility
        """
        self.params = params
        self.rng = np.random.default_rng(seed)

    def simulate_terminal(
        self,
        t: float,
        n_paths: int,
        risk_neutral: bool = True,
    ) -> np.ndarray:
        """
        Simulate terminal stock prices at time t.

        Args:
            t: Time to maturity (in years)
            n_paths: Number of simulation paths
            risk_neutral: If True, use risk-free rate as drift

        Returns:
            Array of terminal stock prices with shape (n_paths,)
        """
        drift = self.params.r if risk_neutral else self.params.mu
        sigma = self.params.sigma

        # Generate standard normal random variables
        z = self.rng.standard_normal(n_paths)

        # Exact GBM solution: S(t) = S(0) * exp((μ - σ²/2)t + σ√t * Z)
        st = self.params.s0 * np.exp(
            (drift - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z
        )

        return st

    def simulate_paths(
        self,
        t: float,
        n_steps: int,
        n_paths: int,
        risk_neutral: bool = True,
    ) -> np.ndarray:
        """
        Simulate full stock price paths from time 0 to t.

        Args:
            t: Total time horizon (in years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            risk_neutral: If True, use risk-free rate as drift

        Returns:
            Array of stock price paths with shape (n_paths, n_steps + 1)
            First column is S(0), last column is S(t)
        """
        drift = self.params.r if risk_neutral else self.params.mu
        sigma = self.params.sigma
        dt = t / n_steps

        # Pre-allocate paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.params.s0

        # Generate all random increments at once
        z = self.rng.standard_normal((n_paths, n_steps))

        # Compute log returns and cumulate
        log_returns = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

        # Cumulative sum of log returns
        paths[:, 1:] = self.params.s0 * np.exp(np.cumsum(log_returns, axis=1))

        return paths

    def get_time_grid(self, t: float, n_steps: int) -> np.ndarray:
        """
        Get the time grid for path simulation.

        Args:
            t: Total time horizon
            n_steps: Number of time steps

        Returns:
            Array of time points with shape (n_steps + 1,)
        """
        return np.linspace(0, t, n_steps + 1)
