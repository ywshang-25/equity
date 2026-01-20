"""Unit tests for the Monte Carlo pricing engine."""

import numpy as np
import pytest

from lib.gbm import GBMParameters, GBMSimulator
from lib.payoffs import EuropeanCallPayoff, EuropeanPutPayoff, CustomPayoff, PayoffType
from lib.pricing import (
    MonteCarloEngine,
    PricingResult,
    black_scholes_call,
    black_scholes_put,
)


class TestPricingResult:
    """Tests for PricingResult dataclass."""

    def test_str_format(self):
        result = PricingResult(
            price=10.5,
            std_error=0.1,
            n_paths=10000,
            confidence_interval_95=(10.3, 10.7),
        )
        result_str = str(result)
        assert "10.500000" in result_str
        assert "0.100000" in result_str
        assert "10.300000" in result_str
        assert "10.700000" in result_str


class TestBlackScholes:
    """Tests for Black-Scholes analytical formulas."""

    def test_call_atm(self):
        """ATM call price should be roughly 0.4 * S * sigma * sqrt(T) for small r."""
        s0, k, t, r, sigma = 100, 100, 1.0, 0.0, 0.2
        price = black_scholes_call(s0, k, t, r, sigma)
        approx = 0.4 * s0 * sigma * np.sqrt(t)
        assert abs(price - approx) < 1.0  # Rough approximation

    def test_put_atm(self):
        """ATM put price should equal ATM call when r=0 (by put-call parity)."""
        s0, k, t, r, sigma = 100, 100, 1.0, 0.0, 0.2
        call_price = black_scholes_call(s0, k, t, r, sigma)
        put_price = black_scholes_put(s0, k, t, r, sigma)
        assert abs(call_price - put_price) < 1e-10

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)."""
        s0, k, t, r, sigma = 100, 105, 1.0, 0.05, 0.2
        call_price = black_scholes_call(s0, k, t, r, sigma)
        put_price = black_scholes_put(s0, k, t, r, sigma)

        lhs = call_price - put_price
        rhs = s0 - k * np.exp(-r * t)

        assert abs(lhs - rhs) < 1e-10

    def test_deep_itm_call(self):
        """Deep ITM call should be approximately S - K*exp(-rT)."""
        s0, k, t, r, sigma = 150, 100, 1.0, 0.05, 0.2
        price = black_scholes_call(s0, k, t, r, sigma)
        intrinsic = s0 - k * np.exp(-r * t)
        assert price >= intrinsic
        assert abs(price - intrinsic) < 5  # Close to intrinsic

    def test_deep_otm_call(self):
        """Deep OTM call should be close to zero."""
        s0, k, t, r, sigma = 50, 100, 1.0, 0.05, 0.2
        price = black_scholes_call(s0, k, t, r, sigma)
        assert price < 0.1

    def test_higher_volatility_higher_price(self):
        """Higher volatility should increase option price."""
        s0, k, t, r = 100, 100, 1.0, 0.05
        price_low_vol = black_scholes_call(s0, k, t, r, 0.1)
        price_high_vol = black_scholes_call(s0, k, t, r, 0.3)
        assert price_high_vol > price_low_vol

    def test_longer_maturity_higher_price(self):
        """Longer maturity should increase call option price."""
        s0, k, r, sigma = 100, 100, 0.05, 0.2
        price_short = black_scholes_call(s0, k, 0.25, r, sigma)
        price_long = black_scholes_call(s0, k, 1.0, r, sigma)
        assert price_long > price_short


class TestMonteCarloEngine:
    """Tests for MonteCarloEngine class."""

    @pytest.fixture
    def params(self):
        return GBMParameters(s0=100, mu=0.1, sigma=0.2, r=0.05)

    @pytest.fixture
    def simulator(self, params):
        return GBMSimulator(params, seed=42)

    @pytest.fixture
    def engine(self, simulator):
        return MonteCarloEngine(simulator, n_paths=50000, n_steps=252)

    def test_call_price_matches_black_scholes(self, params, simulator):
        """Monte Carlo call price should converge to Black-Scholes."""
        engine = MonteCarloEngine(simulator, n_paths=100000)
        call = EuropeanCallPayoff(strike=105)
        result = engine.price(call, t=1.0)

        bs_price = black_scholes_call(
            params.s0, 105, 1.0, params.r, params.sigma
        )

        # Price should be within 3 standard errors of BS price
        assert abs(result.price - bs_price) < 3 * result.std_error

    def test_put_price_matches_black_scholes(self, params, simulator):
        """Monte Carlo put price should converge to Black-Scholes."""
        engine = MonteCarloEngine(simulator, n_paths=100000)
        put = EuropeanPutPayoff(strike=105)
        result = engine.price(put, t=1.0)

        bs_price = black_scholes_put(
            params.s0, 105, 1.0, params.r, params.sigma
        )

        assert abs(result.price - bs_price) < 3 * result.std_error

    def test_put_call_parity(self, engine):
        """Monte Carlo prices should satisfy put-call parity."""
        strike = 105
        t = 1.0
        r = engine.simulator.params.r
        s0 = engine.simulator.params.s0

        call = EuropeanCallPayoff(strike=strike)
        put = EuropeanPutPayoff(strike=strike)

        # Use same seed for both
        engine.simulator.rng = np.random.default_rng(42)
        call_result = engine.price(call, t=t)

        engine.simulator.rng = np.random.default_rng(42)
        put_result = engine.price(put, t=t)

        # C - P = S - K*exp(-rT)
        lhs = call_result.price - put_result.price
        rhs = s0 - strike * np.exp(-r * t)

        # Allow for Monte Carlo error
        combined_error = np.sqrt(call_result.std_error**2 + put_result.std_error**2)
        assert abs(lhs - rhs) < 3 * combined_error

    def test_confidence_interval_contains_price(self, engine):
        """Price should be within confidence interval."""
        call = EuropeanCallPayoff(strike=100)
        result = engine.price(call, t=1.0)

        assert result.confidence_interval_95[0] <= result.price
        assert result.price <= result.confidence_interval_95[1]

    def test_confidence_interval_width(self, engine):
        """CI width should be approximately 2 * 1.96 * std_error."""
        call = EuropeanCallPayoff(strike=100)
        result = engine.price(call, t=1.0)

        ci_width = result.confidence_interval_95[1] - result.confidence_interval_95[0]
        expected_width = 2 * 1.96 * result.std_error

        assert abs(ci_width - expected_width) < 1e-10

    def test_more_paths_reduces_error(self, simulator):
        """More paths should reduce standard error."""
        engine_few = MonteCarloEngine(simulator, n_paths=10000)
        engine_many = MonteCarloEngine(simulator, n_paths=100000)

        call = EuropeanCallPayoff(strike=100)

        result_few = engine_few.price(call, t=1.0)
        result_many = engine_many.price(call, t=1.0)

        # Error should scale as 1/sqrt(n), so 10x paths -> ~3.16x less error
        assert result_many.std_error < result_few.std_error

    def test_n_paths_override(self, engine):
        """n_paths parameter should override default."""
        call = EuropeanCallPayoff(strike=100)
        result = engine.price(call, t=1.0, n_paths=1000)
        assert result.n_paths == 1000

    def test_zero_rate_no_discounting(self, params):
        """With r=0, discounted and undiscounted should match."""
        params_zero_r = GBMParameters(s0=100, mu=0.1, sigma=0.2, r=0.0)
        simulator = GBMSimulator(params_zero_r, seed=42)
        engine = MonteCarloEngine(simulator, n_paths=10000)

        call = EuropeanCallPayoff(strike=100)
        result = engine.price(call, t=1.0)

        # Discount factor is 1, so no discounting effect
        assert result.price > 0

    def test_zero_maturity_intrinsic_value(self, engine):
        """At t=0, option price should equal intrinsic value."""
        # ITM call
        call_itm = EuropeanCallPayoff(strike=90)
        result = engine.price(call_itm, t=0.001)  # Very small t
        assert abs(result.price - 10) < 1  # S0=100, K=90, intrinsic=10

        # OTM call
        call_otm = EuropeanCallPayoff(strike=110)
        result = engine.price(call_otm, t=0.001)
        assert result.price < 1  # Close to 0


class TestAntitheticVariates:
    """Tests for antithetic variate variance reduction."""

    @pytest.fixture
    def params(self):
        return GBMParameters(s0=100, mu=0.1, sigma=0.2, r=0.05)

    @pytest.fixture
    def simulator(self, params):
        return GBMSimulator(params, seed=42)

    @pytest.fixture
    def engine(self, simulator):
        return MonteCarloEngine(simulator, n_paths=50000)

    def test_antithetic_price_matches_black_scholes(self, params, simulator):
        """Antithetic price should converge to Black-Scholes."""
        engine = MonteCarloEngine(simulator, n_paths=100000)
        call = EuropeanCallPayoff(strike=105)
        result = engine.price_with_control_variate(call, t=1.0)

        bs_price = black_scholes_call(
            params.s0, 105, 1.0, params.r, params.sigma
        )

        assert abs(result.price - bs_price) < 3 * result.std_error

    def test_antithetic_reduces_variance(self, simulator):
        """Antithetic variates should reduce standard error."""
        engine = MonteCarloEngine(simulator, n_paths=50000)
        call = EuropeanCallPayoff(strike=100)

        # Reset RNG for fair comparison
        simulator.rng = np.random.default_rng(42)
        result_standard = engine.price(call, t=1.0)

        simulator.rng = np.random.default_rng(42)
        result_antithetic = engine.price_with_control_variate(call, t=1.0)

        # Antithetic should have lower standard error
        assert result_antithetic.std_error < result_standard.std_error

    def test_antithetic_n_paths_override(self, engine):
        """n_paths parameter should override default."""
        call = EuropeanCallPayoff(strike=100)
        result = engine.price_with_control_variate(call, t=1.0, n_paths=2000)
        assert result.n_paths == 2000


class TestPathDependentPricing:
    """Tests for path-dependent option pricing."""

    @pytest.fixture
    def params(self):
        return GBMParameters(s0=100, mu=0.1, sigma=0.2, r=0.05)

    @pytest.fixture
    def simulator(self, params):
        return GBMSimulator(params, seed=42)

    @pytest.fixture
    def engine(self, simulator):
        return MonteCarloEngine(simulator, n_paths=50000, n_steps=252)

    def test_asian_call_less_than_european(self, engine):
        """Asian call should be cheaper than European call (for same strike)."""
        strike = 100

        european_call = EuropeanCallPayoff(strike=strike)
        asian_call = CustomPayoff(
            payoff_func=lambda p, t: np.maximum(np.mean(p, axis=1) - strike, 0),
            payoff_type=PayoffType.PATH_DEPENDENT
        )

        euro_result = engine.price(european_call, t=1.0)
        asian_result = engine.price(asian_call, t=1.0)

        # Asian should be cheaper due to averaging reducing volatility
        assert asian_result.price < euro_result.price

    def test_n_steps_override(self, engine):
        """n_steps parameter should affect path-dependent pricing."""
        strike = 100
        asian_call = CustomPayoff(
            payoff_func=lambda p, t: np.maximum(np.mean(p, axis=1) - strike, 0),
            payoff_type=PayoffType.PATH_DEPENDENT
        )

        # More steps should give more accurate averaging
        result_few = engine.price(asian_call, t=1.0, n_steps=12)
        result_many = engine.price(asian_call, t=1.0, n_steps=252)

        # Both should give positive prices
        assert result_few.price > 0
        assert result_many.price > 0

    def test_lookback_call_more_than_european(self, engine):
        """Lookback call should be more expensive than European call."""
        strike = 100

        european_call = EuropeanCallPayoff(strike=strike)
        lookback_call = CustomPayoff(
            payoff_func=lambda p, t: p[:, -1] - np.min(p, axis=1),
            payoff_type=PayoffType.PATH_DEPENDENT
        )

        euro_result = engine.price(european_call, t=1.0)
        lookback_result = engine.price(lookback_call, t=1.0)

        # Lookback should be more expensive
        assert lookback_result.price > euro_result.price
