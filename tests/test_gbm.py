"""Unit tests for the GBM simulation engine."""

import numpy as np
import pytest

from lib.gbm import GBMParameters, GBMSimulator


class TestGBMParameters:
    """Tests for GBMParameters dataclass."""

    def test_valid_parameters(self):
        params = GBMParameters(s0=100, mu=0.1, sigma=0.2, r=0.05)
        assert params.s0 == 100
        assert params.mu == 0.1
        assert params.sigma == 0.2
        assert params.r == 0.05

    def test_default_risk_free_rate(self):
        params = GBMParameters(s0=100, mu=0.1, sigma=0.2)
        assert params.r == 0.0

    def test_negative_initial_price_raises(self):
        with pytest.raises(ValueError, match="Initial stock price must be positive"):
            GBMParameters(s0=-100, mu=0.1, sigma=0.2)

    def test_zero_initial_price_raises(self):
        with pytest.raises(ValueError, match="Initial stock price must be positive"):
            GBMParameters(s0=0, mu=0.1, sigma=0.2)

    def test_negative_volatility_raises(self):
        with pytest.raises(ValueError, match="Volatility cannot be negative"):
            GBMParameters(s0=100, mu=0.1, sigma=-0.2)

    def test_zero_volatility_allowed(self):
        params = GBMParameters(s0=100, mu=0.1, sigma=0)
        assert params.sigma == 0


class TestGBMSimulator:
    """Tests for GBMSimulator class."""

    @pytest.fixture
    def params(self):
        return GBMParameters(s0=100, mu=0.1, sigma=0.2, r=0.05)

    @pytest.fixture
    def simulator(self, params):
        return GBMSimulator(params, seed=42)

    def test_reproducibility_with_seed(self, params):
        sim1 = GBMSimulator(params, seed=123)
        sim2 = GBMSimulator(params, seed=123)

        result1 = sim1.simulate_terminal(t=1.0, n_paths=100)
        result2 = sim2.simulate_terminal(t=1.0, n_paths=100)

        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_different_results(self, params):
        sim1 = GBMSimulator(params, seed=123)
        sim2 = GBMSimulator(params, seed=456)

        result1 = sim1.simulate_terminal(t=1.0, n_paths=100)
        result2 = sim2.simulate_terminal(t=1.0, n_paths=100)

        assert not np.allclose(result1, result2)

    def test_simulate_terminal_shape(self, simulator):
        result = simulator.simulate_terminal(t=1.0, n_paths=1000)
        assert result.shape == (1000,)

    def test_simulate_terminal_positive_prices(self, simulator):
        result = simulator.simulate_terminal(t=1.0, n_paths=10000)
        assert np.all(result > 0)

    def test_simulate_terminal_mean_risk_neutral(self, params):
        """Under risk-neutral measure, E[S(T)] = S(0) * exp(r*T)."""
        simulator = GBMSimulator(params, seed=42)
        result = simulator.simulate_terminal(t=1.0, n_paths=100000, risk_neutral=True)

        expected_mean = params.s0 * np.exp(params.r * 1.0)
        actual_mean = np.mean(result)

        # Allow 1% relative error due to Monte Carlo variance
        assert abs(actual_mean - expected_mean) / expected_mean < 0.01

    def test_simulate_terminal_mean_real_world(self, params):
        """Under real-world measure, E[S(T)] = S(0) * exp(mu*T)."""
        simulator = GBMSimulator(params, seed=42)
        result = simulator.simulate_terminal(t=1.0, n_paths=100000, risk_neutral=False)

        expected_mean = params.s0 * np.exp(params.mu * 1.0)
        actual_mean = np.mean(result)

        assert abs(actual_mean - expected_mean) / expected_mean < 0.01

    def test_simulate_paths_shape(self, simulator):
        result = simulator.simulate_paths(t=1.0, n_steps=252, n_paths=100)
        assert result.shape == (100, 253)  # n_paths x (n_steps + 1)

    def test_simulate_paths_initial_price(self, simulator, params):
        result = simulator.simulate_paths(t=1.0, n_steps=252, n_paths=100)
        np.testing.assert_array_equal(result[:, 0], params.s0)

    def test_simulate_paths_positive_prices(self, simulator):
        result = simulator.simulate_paths(t=1.0, n_steps=252, n_paths=1000)
        assert np.all(result > 0)

    def test_simulate_paths_terminal_matches_terminal_sim(self, params):
        """Terminal prices from path simulation should have same distribution."""
        sim1 = GBMSimulator(params, seed=42)
        sim2 = GBMSimulator(params, seed=42)

        terminal_direct = sim1.simulate_terminal(t=1.0, n_paths=10000)
        paths = sim2.simulate_paths(t=1.0, n_steps=1, n_paths=10000)
        terminal_from_path = paths[:, -1]

        # With same seed and 1 step, should be identical
        np.testing.assert_array_almost_equal(terminal_direct, terminal_from_path)

    def test_get_time_grid(self, simulator):
        time_grid = simulator.get_time_grid(t=1.0, n_steps=12)

        assert len(time_grid) == 13  # n_steps + 1
        assert time_grid[0] == 0.0
        assert time_grid[-1] == 1.0
        np.testing.assert_array_almost_equal(
            time_grid, np.linspace(0, 1, 13)
        )

    def test_zero_volatility_deterministic(self, params):
        """With zero volatility, paths should be deterministic."""
        params_zero_vol = GBMParameters(s0=100, mu=0.1, sigma=0, r=0.05)
        simulator = GBMSimulator(params_zero_vol, seed=42)

        result = simulator.simulate_terminal(t=1.0, n_paths=100, risk_neutral=True)

        expected = 100 * np.exp(0.05 * 1.0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_longer_maturity_higher_variance(self, simulator):
        """Longer maturity should result in higher variance of terminal prices."""
        result_short = simulator.simulate_terminal(t=0.25, n_paths=10000)

        # Reset RNG for fair comparison
        simulator.rng = np.random.default_rng(42)
        result_long = simulator.simulate_terminal(t=1.0, n_paths=10000)

        assert np.var(result_long) > np.var(result_short)
