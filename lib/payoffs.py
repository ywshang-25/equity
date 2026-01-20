"""
Payoff functions for equity derivatives.

This module provides an extensible framework for defining derivative payoffs.
To create a custom payoff, subclass the Payoff base class and implement
the evaluate() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np


class PayoffType(Enum):
    """Type of payoff based on path dependency."""

    TERMINAL = "terminal"  # Payoff depends only on S(T)
    PATH_DEPENDENT = "path_dependent"  # Payoff depends on full path


class Payoff(ABC):
    """
    Abstract base class for derivative payoffs.

    To implement a custom payoff (including exotic options), subclass this
    and implement the evaluate() method.

    Attributes:
        payoff_type: Indicates whether payoff is terminal or path-dependent
    """

    payoff_type: PayoffType = PayoffType.TERMINAL

    @abstractmethod
    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate the payoff for given price paths.

        Args:
            paths: Stock price paths with shape (n_paths,) for terminal payoffs
                   or (n_paths, n_steps + 1) for path-dependent payoffs
            time_grid: Time points corresponding to path columns (for path-dependent)

        Returns:
            Array of payoff values with shape (n_paths,)
        """
        pass


@dataclass
class EuropeanCallPayoff(Payoff):
    """
    European call option payoff: max(S(T) - K, 0)

    Attributes:
        strike: Strike price K
    """

    strike: float
    payoff_type: PayoffType = PayoffType.TERMINAL

    def __post_init__(self):
        if self.strike < 0:
            raise ValueError("Strike price cannot be negative")

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate call payoff: max(S(T) - K, 0)"""
        # Handle both terminal prices (1D) and full paths (2D)
        terminal_prices = paths if paths.ndim == 1 else paths[:, -1]
        return np.maximum(terminal_prices - self.strike, 0.0)


@dataclass
class EuropeanPutPayoff(Payoff):
    """
    European put option payoff: max(K - S(T), 0)

    Attributes:
        strike: Strike price K
    """

    strike: float
    payoff_type: PayoffType = PayoffType.TERMINAL

    def __post_init__(self):
        if self.strike < 0:
            raise ValueError("Strike price cannot be negative")

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate put payoff: max(K - S(T), 0)"""
        # Handle both terminal prices (1D) and full paths (2D)
        terminal_prices = paths if paths.ndim == 1 else paths[:, -1]
        return np.maximum(self.strike - terminal_prices, 0.0)


class CustomPayoff(Payoff):
    """
    Custom payoff defined by a user-provided function.

    This allows quick prototyping of exotic payoffs without subclassing.

    Example:
        # Digital call paying 1 if S(T) > K
        digital_call = CustomPayoff(
            payoff_func=lambda paths, tg: (paths[:, -1] > 100).astype(float),
            payoff_type=PayoffType.TERMINAL
        )
    """

    def __init__(
        self,
        payoff_func: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        payoff_type: PayoffType = PayoffType.TERMINAL,
    ):
        """
        Initialize custom payoff.

        Args:
            payoff_func: Function taking (paths, time_grid) and returning payoffs
            payoff_type: Whether payoff is terminal or path-dependent
        """
        self._payoff_func = payoff_func
        self.payoff_type = payoff_type

    def evaluate(
        self,
        paths: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the custom payoff function."""
        return self._payoff_func(paths, time_grid)
