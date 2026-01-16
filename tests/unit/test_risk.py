"""
Unit tests for risk management components.
"""

import pytest
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.risk.circuit_breaker import CircuitBreaker, CircuitState
from src.risk.position_sizer import PositionSizer, calculate_position_size
from src.risk.risk_manager import RiskManager
from src.config.settings import RiskConfig


class TestCircuitBreaker:
    """Tests for circuit breaker."""

    def test_initial_state(self):
        """Circuit breaker should start in closed state."""
        cb = CircuitBreaker()
        assert cb.is_trading_allowed()
        state = cb.get_state()
        assert state.state == CircuitState.CLOSED
        assert state.consecutive_losses == 0

    def test_consecutive_losses_trigger(self):
        """Circuit breaker should open after consecutive losses."""
        cb = CircuitBreaker(consecutive_loss_limit=3, cooldown_hours=1)

        cb.record_loss()
        assert cb.is_trading_allowed()

        cb.record_loss()
        assert cb.is_trading_allowed()

        cb.record_loss()  # Third loss triggers circuit breaker
        assert not cb.is_trading_allowed()

        state = cb.get_state()
        assert state.state == CircuitState.OPEN
        assert state.consecutive_losses == 3

    def test_win_resets_losses(self):
        """A win should reset consecutive loss counter."""
        cb = CircuitBreaker(consecutive_loss_limit=3)

        cb.record_loss()
        cb.record_loss()
        assert cb.get_state().consecutive_losses == 2

        cb.record_win()
        assert cb.get_state().consecutive_losses == 0
        assert cb.is_trading_allowed()

    def test_emergency_stop(self):
        """Emergency stop should transition to emergency state."""
        cb = CircuitBreaker()

        cb.trigger_emergency_stop("Max drawdown exceeded")

        assert not cb.is_trading_allowed()
        state = cb.get_state()
        assert state.state == CircuitState.EMERGENCY
        assert state.trigger_reason == "Max drawdown exceeded"

    def test_reset(self):
        """Reset should return to closed state."""
        cb = CircuitBreaker(consecutive_loss_limit=2)

        cb.record_loss()
        cb.record_loss()  # Triggers open

        cb.reset()
        assert cb.is_trading_allowed()
        assert cb.get_state().consecutive_losses == 0

    def test_cooldown_time_calculation(self):
        """Should correctly calculate time until ready."""
        cb = CircuitBreaker(consecutive_loss_limit=1, cooldown_hours=2)

        cb.record_loss()  # Triggers cooldown

        time_until = cb.time_until_ready()
        assert time_until is not None
        # Should be close to 2 hours
        assert time_until.total_seconds() > 7000  # >1.9 hours

    def test_serialization(self):
        """Circuit breaker state should serialize/deserialize correctly."""
        cb = CircuitBreaker(consecutive_loss_limit=3)
        cb.record_loss()
        cb.record_loss()

        data = cb.to_dict()

        cb2 = CircuitBreaker()
        cb2.from_dict(data)

        assert cb2.get_state().consecutive_losses == 2


class TestPositionSizer:
    """Tests for position sizer."""

    def test_max_position_calculation(self):
        """Should calculate max position correctly."""
        sizer = PositionSizer(max_position_percent=5.0, max_position_usd=500.0)

        # 5% of $5000 = $250 (less than $500 cap)
        assert sizer.calculate_max_position(5000.0) == 250.0

        # 5% of $20000 = $1000 (capped at $500)
        assert sizer.calculate_max_position(20000.0) == 500.0

    def test_risk_based_sizing(self):
        """Should size positions based on risk."""
        sizer = PositionSizer(
            max_position_percent=10.0,
            max_position_usd=1000.0,
            risk_per_trade_percent=1.0
        )

        # $10000 capital, 1% risk = $100
        # Entry at $100, stop at $95 = 5% stop distance
        # Position = $100 / 0.05 = $2000 (capped at 10% = $1000)
        result = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=100.0,
            stop_loss_price=95.0
        )

        assert result.size_usd == 1000.0  # Capped at max
        assert result.stop_loss_distance_percent == 5.0

    def test_signal_strength_adjustment(self):
        """Should adjust position size based on signal strength."""
        sizer = PositionSizer(max_position_percent=10.0, max_position_usd=1000.0)

        # Full strength
        result_full = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            signal_strength=1.0
        )

        # Half strength
        result_half = sizer.calculate_position_size(
            capital=10000.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            signal_strength=0.5
        )

        assert result_half.size_usd < result_full.size_usd

    def test_kelly_sizing(self):
        """Kelly criterion should calculate appropriate size."""
        sizer = PositionSizer(max_position_percent=10.0, max_position_usd=1000.0)

        # 60% win rate, 2:1 win/loss ratio
        # Kelly = (2 * 0.6 - 0.4) / 2 = 0.4
        # With 1/4 Kelly fraction = 0.1 = 10%
        size = sizer.calculate_kelly_size(
            capital=10000.0,
            win_rate=0.6,
            avg_win=200.0,
            avg_loss=100.0,
            fraction=0.25
        )

        assert size > 0
        assert size <= 1000.0  # Should be capped

    def test_volatility_adjustment(self):
        """Should reduce position in high volatility."""
        sizer = PositionSizer(max_position_percent=10.0, max_position_usd=1000.0)

        base_size = 1000.0

        # High volatility (2x normal)
        adjusted = sizer.adjust_for_volatility(
            base_size=base_size,
            current_volatility=20.0,
            average_volatility=10.0
        )

        assert adjusted < base_size

    def test_convenience_function(self):
        """Test calculate_position_size convenience function."""
        size = calculate_position_size(
            capital=10000.0,
            entry_price=100.0,
            stop_loss_percent=5.0,
            max_position_percent=10.0
        )

        assert size > 0
        assert size <= 1000.0


class TestRiskManager:
    """Tests for risk manager."""

    @pytest.fixture
    def risk_config(self):
        """Create test risk config."""
        return RiskConfig(
            max_position_size_usd=500.0,
            max_position_percent=5.0,
            max_total_exposure_percent=20.0,
            stop_loss_percent=5.0,
            take_profit_percent=10.0,
            max_daily_trades=5,
            max_daily_loss_percent=3.0,
            circuit_breaker_consecutive_losses=3,
            circuit_breaker_cooldown_hours=4,
            max_drawdown_percent=10.0
        )

    def test_can_trade_initial(self, risk_config):
        """Should allow trading initially."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        check = rm.can_trade()
        assert check.allowed

    def test_daily_trade_limit(self, risk_config):
        """Should block trading after daily limit."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        # Record 5 trades
        for _ in range(5):
            rm.record_trade_result(pnl=10.0, pair="BTC/USD")

        check = rm.can_trade()
        assert not check.allowed
        assert "trade limit" in check.reason.lower()

    def test_daily_loss_limit(self, risk_config):
        """Should block trading after daily loss limit."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        # 3% of $10000 = $300 max daily loss
        rm.record_trade_result(pnl=-350.0, pair="BTC/USD")

        check = rm.can_trade()
        assert not check.allowed
        assert "loss limit" in check.reason.lower()

    def test_position_size_check(self, risk_config):
        """Should approve and adjust position sizes."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        # Request $600 (above $500 max)
        check = rm.check_position_size(600.0, "BTC/USD")

        assert check.allowed
        assert check.adjusted_size == 500.0  # Capped at max

    def test_exposure_limit(self, risk_config):
        """Should block when exposure limit reached."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        # Update exposure to max (20% of $10000 = $2000)
        rm.update_exposure(2000.0)

        check = rm.can_trade()
        assert not check.allowed
        assert "exposure" in check.reason.lower()

    def test_circuit_breaker_integration(self, risk_config):
        """Circuit breaker should block trading after consecutive losses."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        # Record 3 consecutive losses
        for _ in range(3):
            rm.record_trade_result(pnl=-10.0, pair="BTC/USD")

        check = rm.can_trade()
        assert not check.allowed
        assert "circuit breaker" in check.reason.lower()

    def test_emergency_exit_detection(self, risk_config):
        """Should detect when emergency exit is needed."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        # Trigger via circuit breaker
        rm.circuit_breaker.trigger_emergency_stop("Test emergency")

        should_exit, reason = rm.should_emergency_exit()
        assert should_exit
        assert "emergency" in reason.lower()

    def test_get_limits(self, risk_config):
        """Should return current risk limits."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)

        rm.record_trade_result(pnl=-50.0, pair="BTC/USD")

        limits = rm.get_limits()

        assert limits.daily_trades_remaining == 4
        assert limits.daily_loss_remaining == 250.0  # 300 - 50
        assert limits.max_position_size == 500.0

    def test_serialization(self, risk_config):
        """Risk manager state should serialize/deserialize."""
        rm = RiskManager(config=risk_config)
        rm.set_capital(10000.0)
        rm.record_trade_result(pnl=-50.0, pair="BTC/USD")

        data = rm.to_dict()

        rm2 = RiskManager(config=risk_config)
        rm2.from_dict(data)

        assert rm2._total_capital == 10000.0
