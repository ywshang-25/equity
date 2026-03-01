"""Unit tests for the payoff functions."""

import numpy as np
import pytest

from lib.payoffs import (
    Payoff,
    PayoffType,
    EuropeanCallPayoff,
    EuropeanPutPayoff,
    AmericanPayoff,
    AmericanCallPayoff,
    AmericanPutPayoff,
    CustomPayoff,
)


class TestEuropeanCallPayoff:
    """Tests for European call payoff."""

    def test_payoff_type(self):
        call = EuropeanCallPayoff(strike=100)
        assert call.payoff_type == PayoffType.TERMINAL

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="Strike price cannot be negative"):
            EuropeanCallPayoff(strike=-100)

    def test_zero_strike_allowed(self):
        call = EuropeanCallPayoff(strike=0)
        assert call.strike == 0

    def test_in_the_money(self):
        call = EuropeanCallPayoff(strike=100)
        prices = np.array([110, 120, 150])
        payoffs = call.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [10, 20, 50])

    def test_out_of_the_money(self):
        call = EuropeanCallPayoff(strike=100)
        prices = np.array([90, 80, 50])
        payoffs = call.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [0, 0, 0])

    def test_at_the_money(self):
        call = EuropeanCallPayoff(strike=100)
        prices = np.array([100])
        payoffs = call.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [0])

    def test_mixed_moneyness(self):
        call = EuropeanCallPayoff(strike=100)
        prices = np.array([80, 100, 120])
        payoffs = call.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [0, 0, 20])

    def test_2d_paths_uses_terminal(self):
        """When given 2D paths, should use terminal column."""
        call = EuropeanCallPayoff(strike=100)
        paths = np.array([
            [100, 95, 110],  # Terminal: 110
            [100, 105, 90],  # Terminal: 90
            [100, 102, 100], # Terminal: 100
        ])
        payoffs = call.evaluate(paths)
        np.testing.assert_array_equal(payoffs, [10, 0, 0])

    def test_single_path(self):
        call = EuropeanCallPayoff(strike=100)
        prices = np.array([120])
        payoffs = call.evaluate(prices)
        assert payoffs[0] == 20


class TestEuropeanPutPayoff:
    """Tests for European put payoff."""

    def test_payoff_type(self):
        put = EuropeanPutPayoff(strike=100)
        assert put.payoff_type == PayoffType.TERMINAL

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="Strike price cannot be negative"):
            EuropeanPutPayoff(strike=-100)

    def test_in_the_money(self):
        put = EuropeanPutPayoff(strike=100)
        prices = np.array([90, 80, 50])
        payoffs = put.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [10, 20, 50])

    def test_out_of_the_money(self):
        put = EuropeanPutPayoff(strike=100)
        prices = np.array([110, 120, 150])
        payoffs = put.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [0, 0, 0])

    def test_at_the_money(self):
        put = EuropeanPutPayoff(strike=100)
        prices = np.array([100])
        payoffs = put.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [0])

    def test_mixed_moneyness(self):
        put = EuropeanPutPayoff(strike=100)
        prices = np.array([80, 100, 120])
        payoffs = put.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [20, 0, 0])

    def test_2d_paths_uses_terminal(self):
        """When given 2D paths, should use terminal column."""
        put = EuropeanPutPayoff(strike=100)
        paths = np.array([
            [100, 95, 110],  # Terminal: 110
            [100, 105, 90],  # Terminal: 90
            [100, 102, 100], # Terminal: 100
        ])
        payoffs = put.evaluate(paths)
        np.testing.assert_array_equal(payoffs, [0, 10, 0])


class TestPutCallParity:
    """Test put-call parity relationship in payoffs."""

    def test_call_minus_put_equals_intrinsic(self):
        """C(S) - P(S) = S - K for any S."""
        strike = 100
        call = EuropeanCallPayoff(strike=strike)
        put = EuropeanPutPayoff(strike=strike)

        prices = np.array([50, 80, 100, 120, 150])
        call_payoffs = call.evaluate(prices)
        put_payoffs = put.evaluate(prices)

        # C - P = max(S-K, 0) - max(K-S, 0) = S - K
        np.testing.assert_array_almost_equal(
            call_payoffs - put_payoffs,
            prices - strike
        )


class TestCustomPayoff:
    """Tests for CustomPayoff class."""

    def test_terminal_payoff_type(self):
        custom = CustomPayoff(
            payoff_func=lambda p, t: p,
            payoff_type=PayoffType.TERMINAL
        )
        assert custom.payoff_type == PayoffType.TERMINAL

    def test_path_dependent_payoff_type(self):
        custom = CustomPayoff(
            payoff_func=lambda p, t: np.max(p, axis=1),
            payoff_type=PayoffType.PATH_DEPENDENT
        )
        assert custom.payoff_type == PayoffType.PATH_DEPENDENT

    def test_digital_call_payoff(self):
        """Digital call pays 1 if S(T) > K, else 0."""
        strike = 100
        digital_call = CustomPayoff(
            payoff_func=lambda p, t: (p > strike).astype(float),
            payoff_type=PayoffType.TERMINAL
        )

        prices = np.array([90, 100, 110])
        payoffs = digital_call.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [0, 0, 1])

    def test_digital_put_payoff(self):
        """Digital put pays 1 if S(T) < K, else 0."""
        strike = 100
        digital_put = CustomPayoff(
            payoff_func=lambda p, t: (p < strike).astype(float),
            payoff_type=PayoffType.TERMINAL
        )

        prices = np.array([90, 100, 110])
        payoffs = digital_put.evaluate(prices)
        np.testing.assert_array_equal(payoffs, [1, 0, 0])

    def test_asian_call_payoff(self):
        """Asian call payoff based on average price."""
        strike = 100
        asian_call = CustomPayoff(
            payoff_func=lambda p, t: np.maximum(np.mean(p, axis=1) - strike, 0),
            payoff_type=PayoffType.PATH_DEPENDENT
        )

        paths = np.array([
            [100, 110, 120],  # Average: 110 -> Payoff: 10
            [100, 90, 80],    # Average: 90 -> Payoff: 0
            [100, 100, 100],  # Average: 100 -> Payoff: 0
        ])
        payoffs = asian_call.evaluate(paths)
        np.testing.assert_array_equal(payoffs, [10, 0, 0])

    def test_lookback_call_payoff(self):
        """Lookback call payoff: S(T) - min(S)."""
        lookback_call = CustomPayoff(
            payoff_func=lambda p, t: p[:, -1] - np.min(p, axis=1),
            payoff_type=PayoffType.PATH_DEPENDENT
        )

        paths = np.array([
            [100, 90, 110],   # min=90, terminal=110 -> 20
            [100, 80, 95],    # min=80, terminal=95 -> 15
            [100, 100, 100],  # min=100, terminal=100 -> 0
        ])
        payoffs = lookback_call.evaluate(paths)
        np.testing.assert_array_equal(payoffs, [20, 15, 0])

    def test_barrier_knock_out_call(self):
        """Up-and-out call: standard call if max(S) < barrier, else 0."""
        strike = 100
        barrier = 120

        def barrier_payoff(paths, time_grid):
            terminal = paths[:, -1]
            max_price = np.max(paths, axis=1)
            call_payoff = np.maximum(terminal - strike, 0)
            return np.where(max_price < barrier, call_payoff, 0)

        barrier_call = CustomPayoff(
            payoff_func=barrier_payoff,
            payoff_type=PayoffType.PATH_DEPENDENT
        )

        paths = np.array([
            [100, 110, 115],  # max=115 < 120, terminal=115 -> 15
            [100, 125, 110],  # max=125 >= 120 -> knocked out -> 0
            [100, 90, 95],    # max=100 < 120, terminal=95 -> 0 (OTM)
        ])
        payoffs = barrier_call.evaluate(paths)
        np.testing.assert_array_equal(payoffs, [15, 0, 0])

    def test_time_grid_passed_to_payoff(self):
        """Verify time_grid is correctly passed to custom payoff."""
        received_time_grid = []

        def capture_time_grid(paths, time_grid):
            received_time_grid.append(time_grid)
            return np.zeros(len(paths))

        custom = CustomPayoff(
            payoff_func=capture_time_grid,
            payoff_type=PayoffType.PATH_DEPENDENT
        )

        time_grid = np.array([0, 0.5, 1.0])
        paths = np.array([[100, 105, 110]])
        custom.evaluate(paths, time_grid)

        np.testing.assert_array_equal(received_time_grid[0], time_grid)


class TestAmericanCallPayoff:
    """Tests for AmericanCallPayoff."""

    def test_payoff_type(self):
        call = AmericanCallPayoff(strike=100)
        assert call.payoff_type == PayoffType.PATH_DEPENDENT

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="Strike price cannot be negative"):
            AmericanCallPayoff(strike=-10)

    def test_intrinsic_value_itm(self):
        call = AmericanCallPayoff(strike=100)
        S = np.array([110.0, 120.0, 150.0])
        np.testing.assert_array_equal(call.intrinsic_value(S), [10, 20, 50])

    def test_intrinsic_value_otm(self):
        call = AmericanCallPayoff(strike=100)
        S = np.array([80.0, 90.0, 100.0])
        np.testing.assert_array_equal(call.intrinsic_value(S), [0, 0, 0])

    def test_evaluate_uses_terminal_price(self):
        """evaluate() should apply intrinsic_value to the last path column."""
        call = AmericanCallPayoff(strike=100)
        paths = np.array([
            [100, 95, 115],  # Terminal: 115 -> 15
            [100, 110, 90],  # Terminal: 90  -> 0
        ])
        np.testing.assert_array_equal(call.evaluate(paths), [15, 0])

    def test_is_american_payoff_subclass(self):
        assert isinstance(AmericanCallPayoff(strike=100), AmericanPayoff)


class TestAmericanPutPayoff:
    """Tests for AmericanPutPayoff."""

    def test_payoff_type(self):
        put = AmericanPutPayoff(strike=100)
        assert put.payoff_type == PayoffType.PATH_DEPENDENT

    def test_negative_strike_raises(self):
        with pytest.raises(ValueError, match="Strike price cannot be negative"):
            AmericanPutPayoff(strike=-10)

    def test_intrinsic_value_itm(self):
        put = AmericanPutPayoff(strike=100)
        S = np.array([80.0, 90.0, 50.0])
        np.testing.assert_array_equal(put.intrinsic_value(S), [20, 10, 50])

    def test_intrinsic_value_otm(self):
        put = AmericanPutPayoff(strike=100)
        S = np.array([110.0, 120.0, 100.0])
        np.testing.assert_array_equal(put.intrinsic_value(S), [0, 0, 0])

    def test_evaluate_uses_terminal_price(self):
        put = AmericanPutPayoff(strike=100)
        paths = np.array([
            [100, 105, 90],  # Terminal: 90 -> 10
            [100, 95, 110],  # Terminal: 110 -> 0
        ])
        np.testing.assert_array_equal(put.evaluate(paths), [10, 0])

    def test_is_american_payoff_subclass(self):
        assert isinstance(AmericanPutPayoff(strike=100), AmericanPayoff)


class TestAmericanPayoffAbstractClass:
    """Test that AmericanPayoff is properly abstract."""

    def test_cannot_instantiate_without_intrinsic_value(self):
        class IncompleteAmerican(AmericanPayoff):
            pass

        with pytest.raises(TypeError):
            IncompleteAmerican()

    def test_concrete_subclass_works(self):
        class ConcreteAmerican(AmericanPayoff):
            def intrinsic_value(self, S):
                return np.maximum(S - 100, 0)

        payoff = ConcreteAmerican()
        S = np.array([110.0, 90.0])
        np.testing.assert_array_equal(payoff.intrinsic_value(S), [10, 0])


class TestPayoffAbstractClass:
    """Test that Payoff is properly abstract."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Payoff()

    def test_subclass_must_implement_evaluate(self):
        class IncompletePayoff(Payoff):
            pass

        with pytest.raises(TypeError):
            IncompletePayoff()

    def test_subclass_with_evaluate_works(self):
        class ConcretePayoff(Payoff):
            def evaluate(self, paths, time_grid=None):
                return paths

        payoff = ConcretePayoff()
        result = payoff.evaluate(np.array([1, 2, 3]))
        np.testing.assert_array_equal(result, [1, 2, 3])
