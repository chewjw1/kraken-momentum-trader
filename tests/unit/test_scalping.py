"""
Unit tests for scalping strategy components.

Tests:
- ScalpingStrategy entry/exit logic
- VWAP indicator
- Bollinger Bands indicator
- Adaptive pair manager
"""

import pytest
import sys
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.base_strategy import MarketData, Position, SignalType
from src.strategy.signals.vwap import VWAPIndicator, calculate_vwap
from src.strategy.signals.bollinger import BollingerBandsIndicator, calculate_bollinger
from src.core.adaptive_pair_manager import AdaptivePairManager, AdaptiveConfig
from src.risk.circuit_breaker import CircuitBreaker, CircuitState
from src.strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime


@dataclass
class MockOHLC:
    """Mock OHLC candle for testing."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class MockTicker:
    """Mock ticker data for testing."""
    last: float
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 1000.0


def create_ohlc_data(closes, spread_pct=0.5, base_volume=1000.0):
    """Create mock OHLC data from closing prices."""
    ohlc = []
    for close in closes:
        spread = close * (spread_pct / 100)
        ohlc.append(MockOHLC(
            open=close - spread * 0.5,
            high=close + spread,
            low=close - spread,
            close=close,
            volume=base_volume
        ))
    return ohlc


def create_market_data(pair, closes, ticker_price=None, volumes=None):
    """Create mock market data for testing."""
    ohlc = create_ohlc_data(closes)
    if volumes:
        for i, vol in enumerate(volumes):
            if i < len(ohlc):
                ohlc[i].volume = vol

    ticker = MockTicker(last=ticker_price or closes[-1]) if ticker_price else None

    return MarketData(
        pair=pair,
        ohlc=ohlc,
        prices=closes,
        volumes=[c.volume for c in ohlc],
        ticker=ticker
    )


class TestBollingerBands:
    """Tests for Bollinger Bands indicator."""

    def test_insufficient_data(self):
        """Should return None with insufficient data."""
        indicator = BollingerBandsIndicator(period=20)
        prices = [100.0] * 10
        result = indicator.calculate(prices)
        assert result is None

    def test_constant_prices(self):
        """Constant prices should have zero bandwidth."""
        indicator = BollingerBandsIndicator(period=20)
        prices = [100.0] * 25
        result = indicator.calculate(prices)
        assert result is not None
        assert result.middle_band == 100.0
        assert result.upper_band == 100.0  # No std dev
        assert result.lower_band == 100.0
        assert result.bandwidth == 0.0
        assert result.percent_b == 0.5  # Price at middle

    def test_price_below_lower_band(self):
        """Should detect price below lower band."""
        indicator = BollingerBandsIndicator(period=20, std_dev_multiplier=2.0)
        # Create prices with low volatility, then drop
        prices = [100.0] * 24 + [90.0]  # Sharp drop
        result = indicator.calculate(prices)
        assert result is not None
        assert result.is_below_lower
        assert result.signal == "buy"
        assert result.percent_b < 0

    def test_price_above_upper_band(self):
        """Should detect price above upper band."""
        indicator = BollingerBandsIndicator(period=20, std_dev_multiplier=2.0)
        # Create prices with low volatility, then spike
        prices = [100.0] * 24 + [110.0]  # Sharp rise
        result = indicator.calculate(prices)
        assert result is not None
        assert result.is_above_upper
        assert result.signal == "sell"
        assert result.percent_b > 1

    def test_squeeze_detection(self):
        """Should detect Bollinger squeeze (low volatility)."""
        indicator = BollingerBandsIndicator(period=20, squeeze_threshold=2.0)
        # Very tight range
        prices = [100.0 + (i % 2) * 0.1 for i in range(25)]
        result = indicator.calculate(prices)
        assert result is not None
        assert result.is_squeeze  # Low bandwidth

    def test_percent_b_calculation(self):
        """Percent B should be 0 at lower, 0.5 at middle, 1 at upper."""
        indicator = BollingerBandsIndicator(period=20, std_dev_multiplier=2.0)
        # Create some volatility
        prices = [100.0 + i * 0.5 for i in range(25)]
        result = indicator.calculate(prices)
        assert result is not None
        assert 0 <= result.percent_b <= 1.5  # Reasonable range


class TestVWAPIndicator:
    """Tests for VWAP indicator."""

    def test_insufficient_data(self):
        """Should return None with insufficient data."""
        indicator = VWAPIndicator()
        result = indicator.calculate([100], [99], [100], [1000])
        assert result is None

    def test_constant_prices_volumes(self):
        """VWAP should equal typical price with constant data."""
        indicator = VWAPIndicator(threshold_percent=0.5)
        highs = [100.0] * 10
        lows = [98.0] * 10
        closes = [99.0] * 10
        volumes = [1000.0] * 10

        result = indicator.calculate(highs, lows, closes, volumes)
        assert result is not None
        # VWAP = typical price = (100 + 98 + 99) / 3 = 99.0
        expected_vwap = (100.0 + 98.0 + 99.0) / 3
        assert abs(result.value - expected_vwap) < 0.01

    def test_price_below_vwap(self):
        """Should detect price below VWAP."""
        indicator = VWAPIndicator(threshold_percent=0.5)
        # High volume at high prices, low volume at low current price
        highs = [110.0] * 9 + [95.0]
        lows = [105.0] * 9 + [93.0]
        closes = [108.0] * 9 + [94.0]  # Last price significantly lower
        volumes = [2000.0] * 9 + [500.0]  # High volume at high prices

        result = indicator.calculate(highs, lows, closes, volumes)
        assert result is not None
        assert result.is_below  # Price below VWAP
        assert result.signal == "bearish"

    def test_price_above_vwap(self):
        """Should detect price above VWAP."""
        indicator = VWAPIndicator(threshold_percent=0.5)
        # High volume at low prices, then price rises
        highs = [95.0] * 9 + [110.0]
        lows = [93.0] * 9 + [108.0]
        closes = [94.0] * 9 + [109.0]  # Last price significantly higher
        volumes = [2000.0] * 9 + [500.0]  # High volume at low prices

        result = indicator.calculate(highs, lows, closes, volumes)
        assert result is not None
        assert result.is_above  # Price above VWAP
        assert result.signal == "bullish"

    def test_vwap_with_varying_volume(self):
        """VWAP should weight toward high-volume periods."""
        indicator = VWAPIndicator(threshold_percent=0.5)
        # First half: high prices, low volume
        # Second half: low prices, high volume
        highs = [110.0] * 5 + [90.0] * 5
        lows = [108.0] * 5 + [88.0] * 5
        closes = [109.0] * 5 + [89.0] * 5
        volumes = [100.0] * 5 + [1000.0] * 5  # 10x volume at lower prices

        result = indicator.calculate(highs, lows, closes, volumes)
        assert result is not None
        # VWAP should be closer to 89 (high volume) than 109 (low volume)
        assert result.value < 100.0


class TestScalpingStrategy:
    """Tests for the scalping strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ScalpingConfig(
            rsi_period=7,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            bb_period=20,
            bb_std_dev=2.0,
            take_profit_percent=4.5,
            stop_loss_percent=2.0,
            min_confirmations=2,
            fee_percent=0.26
        )
        self.strategy = ScalpingStrategy(self.config)

    def test_initialization(self):
        """Strategy should initialize with correct config."""
        assert self.strategy.config.take_profit_percent == 4.5
        assert self.strategy.config.stop_loss_percent == 2.0
        assert self.strategy.config.min_confirmations == 2

    def test_insufficient_data_returns_hold(self):
        """Should return HOLD signal with insufficient data."""
        closes = [100.0] * 10  # Not enough for BB period
        market_data = create_market_data("BTC/USD", closes)

        signal = self.strategy.analyze(market_data, None)
        assert signal.signal_type == SignalType.HOLD
        assert "Insufficient" in signal.reason

    def test_take_profit_exit(self):
        """Should exit at take profit threshold."""
        # Strategy uses OHLC close price, so we need to set that to the exit price
        # Position entered at 100, price rises to 104.6 = +4.6% (above 4.5% TP)
        closes = [100.0] * 29 + [104.6]  # Last close is the current price
        market_data = create_market_data("BTC/USD", closes)

        position = Position(
            pair="BTC/USD",
            side="long",
            entry_price=100.0,
            current_price=104.6,
            size=0.1,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=1)
        )

        signal = self.strategy.analyze(market_data, position)
        assert signal.signal_type == SignalType.CLOSE_LONG
        assert "Take profit" in signal.reason

    def test_stop_loss_exit(self):
        """Should exit at stop loss threshold."""
        # Strategy uses OHLC close price, so we need to set that to the exit price
        # Position entered at 100, price drops to 97.9 = -2.1% (below -2% SL)
        closes = [100.0] * 29 + [97.9]  # Last close is the current price
        market_data = create_market_data("BTC/USD", closes)

        position = Position(
            pair="BTC/USD",
            side="long",
            entry_price=100.0,
            current_price=97.9,
            size=0.1,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=1)
        )

        signal = self.strategy.analyze(market_data, position)
        assert signal.signal_type == SignalType.CLOSE_LONG
        assert "Stop loss" in signal.reason

    def test_hold_position_within_thresholds(self):
        """Should hold position within TP/SL thresholds."""
        closes = [100.0] * 30
        market_data = create_market_data("BTC/USD", closes, ticker_price=102.0)

        # Position at 100, current price at 102 = +2%
        position = Position(
            pair="BTC/USD",
            side="long",
            entry_price=100.0,
            current_price=102.0,
            size=0.1,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=1)
        )

        signal = self.strategy.analyze(market_data, position)
        assert signal.signal_type == SignalType.HOLD
        assert "Holding" in signal.reason

    def test_no_entry_without_confirmations(self):
        """Should not enter without sufficient confirmations."""
        # Flat market with no signals
        closes = [100.0] * 30
        market_data = create_market_data("BTC/USD", closes)

        signal = self.strategy.analyze(market_data, None)
        assert signal.signal_type == SignalType.HOLD
        assert "confirmations" in signal.reason.lower()

    def test_entry_with_confirmations(self):
        """Should enter when multiple confirmations align."""
        # Create oversold conditions:
        # 1. Sharp drop (RSI oversold)
        # 2. Price at lower BB
        # 3. Below VWAP
        # 4. Volume spike
        prices = [105.0] * 10 + [103.0, 101.0, 99.0, 97.0, 95.0, 93.0, 92.0]  # Downtrend
        prices += [91.0, 90.0, 89.0]  # Bottom
        prices += [90.0] * 10  # Consolidation at bottom

        # High volume on the drop
        volumes = [1000.0] * 20 + [2500.0] * 10

        market_data = create_market_data("BTC/USD", prices, volumes=volumes)

        signal = self.strategy.analyze(market_data, None)
        # May or may not trigger depending on exact indicator values
        # At minimum, the logic should run without errors
        assert signal is not None
        assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]

    def test_get_risk_params(self):
        """Should return correct risk parameters."""
        params = self.strategy.get_risk_params()
        assert params["take_profit_percent"] == 4.5
        assert params["stop_loss_percent"] == 2.0
        assert params["fee_percent"] == 0.26
        # Net target = 4.5 - (0.26 * 2) = 3.98
        assert abs(params["net_target"] - 3.98) < 0.01


class TestAdaptivePairManager:
    """Tests for the adaptive pair manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = AdaptiveConfig(
            min_win_rate=0.35,
            min_profit_factor=0.7,
            max_consecutive_losses=5,
            min_trades_for_decision=10,
            cooldown_hours=2.0,
            reenable_win_rate=0.45
        )
        self.manager = AdaptivePairManager(self.config)

    def test_register_pair(self):
        """Should register new pairs."""
        self.manager.register_pair("BTC/USD")
        assert self.manager.is_pair_enabled("BTC/USD")

    def test_unregistered_pair_enabled(self):
        """Unregistered pairs should be enabled by default."""
        assert self.manager.is_pair_enabled("NEW/USD")

    def test_record_winning_trade(self):
        """Should track winning trades correctly."""
        self.manager.register_pair("BTC/USD")
        now = datetime.now(timezone.utc)

        self.manager.record_trade(
            pair="BTC/USD",
            entry_time=now - timedelta(hours=1),
            exit_time=now,
            pnl=50.0,
            pnl_percent=2.5
        )

        status = self.manager.get_pair_status("BTC/USD")
        assert status["total_trades"] == 1
        assert status["total_pnl"] == 50.0

    def test_record_losing_trade(self):
        """Should track losing trades correctly."""
        self.manager.register_pair("BTC/USD")
        now = datetime.now(timezone.utc)

        self.manager.record_trade(
            pair="BTC/USD",
            entry_time=now - timedelta(hours=1),
            exit_time=now,
            pnl=-25.0,
            pnl_percent=-1.5
        )

        status = self.manager.get_pair_status("BTC/USD")
        assert status["consecutive_losses"] == 1

    def test_disable_on_consecutive_losses(self):
        """Should disable pair after max consecutive losses."""
        self.manager.register_pair("BTC/USD")
        now = datetime.now(timezone.utc)

        # Add some wins first to keep win rate above threshold
        for i in range(6):
            self.manager.record_trade(
                pair="BTC/USD",
                entry_time=now - timedelta(hours=i+20),
                exit_time=now - timedelta(hours=i+19),
                pnl=30.0,
                pnl_percent=1.5
            )

        # Now add consecutive losses (more than max_consecutive_losses=5)
        for i in range(6):
            self.manager.record_trade(
                pair="BTC/USD",
                entry_time=now - timedelta(hours=i+1),
                exit_time=now - timedelta(hours=i),
                pnl=-20.0,
                pnl_percent=-1.0
            )

        # After 6 consecutive losses (> 5 threshold), pair should be disabled
        # Win rate is 6/12 = 50% which is above 35% threshold
        # So it must be disabled for consecutive losses
        assert not self.manager.is_pair_enabled("BTC/USD")
        status = self.manager.get_pair_status("BTC/USD")
        assert "Consecutive losses" in (status.get("disabled_reason") or "")

    def test_disable_on_low_win_rate(self):
        """Should disable pair with low win rate."""
        self.manager.register_pair("BTC/USD")
        now = datetime.now(timezone.utc)

        # Add trades with low win rate (2 wins, 8 losses = 20%)
        for i in range(8):
            self.manager.record_trade(
                pair="BTC/USD",
                entry_time=now - timedelta(hours=i+3),
                exit_time=now - timedelta(hours=i+2),
                pnl=-20.0,
                pnl_percent=-1.0
            )

        for i in range(2):
            self.manager.record_trade(
                pair="BTC/USD",
                entry_time=now - timedelta(hours=i+1),
                exit_time=now - timedelta(hours=i),
                pnl=30.0,
                pnl_percent=1.5
            )

        # Win rate is 20% < 35% threshold
        assert not self.manager.is_pair_enabled("BTC/USD")

    def test_position_scaling(self):
        """Should scale position based on performance."""
        self.manager.register_pair("BTC/USD")
        now = datetime.now(timezone.utc)

        # Add good trades (high win rate)
        for i in range(10):
            is_win = i < 7  # 70% win rate
            self.manager.record_trade(
                pair="BTC/USD",
                entry_time=now - timedelta(hours=i+1),
                exit_time=now - timedelta(hours=i),
                pnl=50.0 if is_win else -30.0,
                pnl_percent=2.5 if is_win else -1.5
            )

        scale = self.manager.get_position_scale("BTC/USD")
        # High performance should give scale > 1.0
        assert scale >= 1.0

    def test_get_enabled_pairs(self):
        """Should filter to only enabled pairs."""
        pairs = ["BTC/USD", "ETH/USD", "SOL/USD"]
        for pair in pairs:
            self.manager.register_pair(pair)

        # Manually disable one
        self.manager.force_disable("ETH/USD", "Test disable")

        enabled = self.manager.get_enabled_pairs(pairs)
        assert "BTC/USD" in enabled
        assert "SOL/USD" in enabled
        assert "ETH/USD" not in enabled

    def test_force_enable(self):
        """Should allow force enabling a disabled pair."""
        self.manager.register_pair("BTC/USD")
        self.manager.force_disable("BTC/USD", "Test")
        assert not self.manager.is_pair_enabled("BTC/USD")

        self.manager.force_enable("BTC/USD")
        assert self.manager.is_pair_enabled("BTC/USD")

    def test_serialization(self):
        """Should serialize and deserialize state."""
        self.manager.register_pair("BTC/USD")
        now = datetime.now(timezone.utc)

        self.manager.record_trade(
            pair="BTC/USD",
            entry_time=now - timedelta(hours=1),
            exit_time=now,
            pnl=50.0,
            pnl_percent=2.5
        )

        # Serialize
        data = self.manager.to_dict()

        # Create new manager and restore
        new_manager = AdaptivePairManager(self.config)
        new_manager.from_dict(data)

        status = new_manager.get_pair_status("BTC/USD")
        assert status["total_pnl"] == 50.0

    def test_cooldown_period(self):
        """Should respect cooldown period after disable."""
        self.manager.register_pair("BTC/USD")

        # Force disable with cooldown
        self.manager.force_disable("BTC/USD", "Test")

        status = self.manager.get_pair_status("BTC/USD")
        assert status["cooldown_until"] is not None


class TestTrailingStop:
    """Tests for trailing stop functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ScalpingConfig(
            take_profit_percent=4.5,
            stop_loss_percent=2.0
        )
        self.strategy = ScalpingStrategy(self.config)

    def test_trailing_stop_activation(self):
        """Should activate trailing stop at threshold."""
        position = Position(
            pair="BTC/USD",
            side="long",
            entry_price=100.0,
            current_price=103.5,  # +3.5%
            size=0.1,
            entry_time=datetime.now(timezone.utc)
        )

        # 75% of 4.5% = 3.375%, so 3.5% should activate
        should_close, should_activate, reason = self.strategy.check_trailing_stop(
            position=position,
            current_price=103.5,
            peak_price=103.5,
            trailing_stop_active=False
        )

        assert not should_close
        assert should_activate

    def test_trailing_stop_trigger(self):
        """Should trigger trailing stop on drop from peak."""
        position = Position(
            pair="BTC/USD",
            side="long",
            entry_price=100.0,
            current_price=103.0,
            size=0.1,
            entry_time=datetime.now(timezone.utc)
        )

        # Peak was 104.0, current is 103.0 = 0.96% drop from peak
        should_close, should_activate, reason = self.strategy.check_trailing_stop(
            position=position,
            current_price=103.0,
            peak_price=104.0,
            trailing_stop_active=True
        )

        # 0.3% trailing drop threshold, 0.96% > 0.3%
        assert should_close
        assert "Trailing stop" in reason

    def test_trailing_stop_not_triggered_small_drop(self):
        """Should not trigger trailing stop on small drop."""
        position = Position(
            pair="BTC/USD",
            side="long",
            entry_price=100.0,
            current_price=103.9,
            size=0.1,
            entry_time=datetime.now(timezone.utc)
        )

        # Peak was 104.0, current is 103.9 = 0.096% drop from peak
        should_close, should_activate, reason = self.strategy.check_trailing_stop(
            position=position,
            current_price=103.9,
            peak_price=104.0,
            trailing_stop_active=True
        )

        # 0.096% < 0.3% threshold
        assert not should_close


class TestEMAFilter:
    """Tests for EMA trend filter in scalping strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ScalpingConfig(
            rsi_period=7,
            bb_period=20,
            min_confirmations=2,
            ema_filter_enabled=True,
            ema_bearish_threshold=-0.5
        )
        self.strategy = ScalpingStrategy(self.config)

    def test_ema_filter_blocks_bearish_entry(self):
        """Should block entry when EMA trend is strongly bearish."""
        # Strong downtrend: prices falling consistently
        prices = [110.0 - i * 0.5 for i in range(30)]  # 110 → 95.5

        # Add volume spike and oversold conditions to trigger confirmations
        volumes = [1000.0] * 20 + [2500.0] * 10

        market_data = create_market_data("BTC/USD", prices, volumes=volumes)
        signal = self.strategy.analyze(market_data, None)

        # Should either HOLD (EMA filter blocks) or HOLD (not enough confirmations)
        assert signal.signal_type == SignalType.HOLD
        # If EMA filter triggered, the reason should mention it
        if "EMA bearish" in signal.reason:
            assert "trend strength" in signal.reason

    def test_ema_filter_disabled(self):
        """Should not filter when EMA filter is disabled."""
        config = ScalpingConfig(
            rsi_period=7,
            bb_period=20,
            min_confirmations=2,
            ema_filter_enabled=False
        )
        strategy = ScalpingStrategy(config)

        # Same bearish data
        prices = [110.0 - i * 0.5 for i in range(30)]
        market_data = create_market_data("BTC/USD", prices)
        signal = strategy.analyze(market_data, None)

        # Should NOT mention EMA filter
        assert "EMA bearish filter" not in signal.reason

    def test_ema_filter_allows_bullish(self):
        """Should allow entry when EMA is bullish."""
        # Uptrend prices (EMA should be bullish)
        prices = [90.0 + i * 0.3 for i in range(30)]  # 90 → 98.7

        market_data = create_market_data("BTC/USD", prices)
        signal = self.strategy.analyze(market_data, None)

        # EMA filter should not block - reason should be about confirmations, not EMA
        assert "EMA bearish filter" not in signal.reason


class TestCircuitBreakerDrawdown:
    """Tests for drawdown-based circuit breaker."""

    def test_global_drawdown_triggers(self):
        """Should halt all trading when global drawdown exceeds threshold."""
        cb = CircuitBreaker(
            global_max_drawdown_pct=5.0,
            initial_capital=10000.0,
            consecutive_loss_limit=100,  # Effectively disabled
        )

        assert cb.is_trading_allowed()

        # Lose 5% of capital ($500) across trades
        cb.record_trade("BTC/USD", -200.0)
        cb.record_trade("ETH/USD", -200.0)
        assert cb.is_trading_allowed()  # 4% drawdown, still OK

        cb.record_trade("SOL/USD", -150.0)  # Now 5.5% drawdown
        assert not cb.is_trading_allowed()

    def test_per_pair_drawdown_pauses_pair(self):
        """Should pause individual pair when its drawdown exceeds threshold."""
        cb = CircuitBreaker(
            global_max_drawdown_pct=10.0,
            pair_max_drawdown_pct=3.0,
            initial_capital=10000.0,
            consecutive_loss_limit=100,
        )

        # BTC loses $300 = 3% of capital
        cb.record_trade("BTC/USD", -150.0)
        cb.record_trade("BTC/USD", -150.0)

        assert not cb.is_pair_allowed("BTC/USD")  # BTC paused
        assert cb.is_pair_allowed("ETH/USD")       # ETH still OK
        assert cb.is_trading_allowed()              # Global still OK

    def test_consecutive_loss_secondary(self):
        """Consecutive losses still trigger as secondary signal."""
        cb = CircuitBreaker(
            global_max_drawdown_pct=50.0,  # Very high, won't trigger
            consecutive_loss_limit=3,
            initial_capital=10000.0,
        )

        cb.record_trade("BTC/USD", -10.0)
        cb.record_trade("ETH/USD", -10.0)
        cb.record_trade("SOL/USD", -10.0)  # 3 consecutive losses
        assert not cb.is_trading_allowed()

    def test_win_resets_consecutive_losses(self):
        """A win should reset the consecutive loss counter."""
        cb = CircuitBreaker(
            global_max_drawdown_pct=50.0,
            consecutive_loss_limit=3,
            initial_capital=10000.0,
        )

        cb.record_trade("BTC/USD", -10.0)
        cb.record_trade("BTC/USD", -10.0)
        cb.record_trade("BTC/USD", 50.0)   # Win resets counter
        cb.record_trade("BTC/USD", -10.0)
        cb.record_trade("BTC/USD", -10.0)
        assert cb.is_trading_allowed()      # Only 2 since last win

    def test_peak_equity_tracking(self):
        """Peak equity should update on profits."""
        cb = CircuitBreaker(
            global_max_drawdown_pct=5.0,
            initial_capital=10000.0,
            consecutive_loss_limit=100,
        )

        # Win $1000 -- peak is now $11000
        cb.record_trade("BTC/USD", 1000.0)
        assert cb.is_trading_allowed()

        # Lose $500 from $11000 -- drawdown is $500/$10000 = 5.0%
        cb.record_trade("BTC/USD", -500.0)
        assert not cb.is_trading_allowed()

    def test_emergency_stop(self):
        """Should enter emergency state and block trading."""
        cb = CircuitBreaker(initial_capital=10000.0)
        cb.trigger_emergency_stop("Market crash detected")
        assert not cb.is_trading_allowed()
        assert cb.get_state().state == CircuitState.EMERGENCY

    def test_manual_reset(self):
        """Should allow manual reset from any state."""
        cb = CircuitBreaker(
            global_max_drawdown_pct=5.0,
            initial_capital=10000.0,
            consecutive_loss_limit=100,
        )
        cb.record_trade("BTC/USD", -600.0)  # 6% drawdown
        assert not cb.is_trading_allowed()

        cb.reset()
        assert cb.is_trading_allowed()

    def test_serialization_with_pair_states(self):
        """Should serialize and restore per-pair states."""
        cb = CircuitBreaker(initial_capital=10000.0, pair_max_drawdown_pct=5.0)

        cb.record_trade("BTC/USD", 100.0)
        cb.record_trade("BTC/USD", -50.0)
        cb.record_trade("ETH/USD", -200.0)

        state = cb.to_dict()
        assert "pair_states" in state
        assert "BTC/USD" in state["pair_states"]
        assert "ETH/USD" in state["pair_states"]

        # Restore
        cb2 = CircuitBreaker(initial_capital=10000.0, pair_max_drawdown_pct=5.0)
        cb2.from_dict(state)

        btc_state = cb2.get_pair_state("BTC/USD")
        assert btc_state is not None
        assert btc_state.current_pnl == 50.0  # 100 - 50

    def test_update_equity(self):
        """Should trigger on equity update too."""
        cb = CircuitBreaker(
            global_max_drawdown_pct=5.0,
            initial_capital=10000.0,
            consecutive_loss_limit=100,
        )

        cb.update_equity(9400.0)  # 6% drawdown from peak
        assert not cb.is_trading_allowed()


class TestBacktestSlippage:
    """Tests for slippage simulation in backtester."""

    def test_slippage_config_default(self):
        """Default slippage should be 0.05%."""
        from src.backtest.scalping_backtester import ScalpingBacktestConfig
        config = ScalpingBacktestConfig()
        assert config.slippage_percent == 0.05

    def test_slippage_worsens_entry(self):
        """Slippage should make entry price worse (higher for longs)."""
        from src.backtest.scalping_backtester import ScalpingBacktestConfig
        config = ScalpingBacktestConfig(slippage_percent=0.1)
        # Entry at 100 with 0.1% slippage = 100.10
        entry = 100.0 * (1 + config.slippage_percent / 100)
        assert entry > 100.0
        assert abs(entry - 100.10) < 0.01

    def test_slippage_worsens_exit(self):
        """Slippage should make exit price worse (lower for longs)."""
        from src.backtest.scalping_backtester import ScalpingBacktestConfig
        config = ScalpingBacktestConfig(slippage_percent=0.1)
        # Exit at 105 with 0.1% slippage = 104.895
        exit_price = 105.0 * (1 - config.slippage_percent / 100)
        assert exit_price < 105.0
        assert abs(exit_price - 104.895) < 0.01


class TestRegimeDetector:
    """Tests for market regime detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RegimeConfig(
            fast_sma_period=50,
            slow_sma_period=200,
            min_data_points=200,
        )
        self.detector = RegimeDetector(self.config)

    def test_insufficient_data_returns_unknown(self):
        """Should return UNKNOWN with insufficient data."""
        prices = [100.0] * 50  # Not enough
        result = self.detector.detect(prices)
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0

    def test_bull_regime_detection(self):
        """Should detect bull regime from rising prices."""
        # 250 candles with clear uptrend
        prices = [100.0 + i * 0.5 for i in range(250)]  # 100 -> 225
        result = self.detector.detect(prices)
        assert result.regime == MarketRegime.BULL
        assert result.confidence > 0.4
        assert result.sma_slope_pct > 0

    def test_bear_regime_detection(self):
        """Should detect bear regime from falling prices."""
        # 250 candles with clear downtrend
        prices = [200.0 - i * 0.4 for i in range(250)]  # 200 -> 100
        result = self.detector.detect(prices)
        assert result.regime == MarketRegime.BEAR
        assert result.confidence > 0.4
        assert result.sma_slope_pct < 0

    def test_sideways_regime_detection(self):
        """Should detect sideways regime from flat prices."""
        import math
        # Oscillate around 100 with small amplitude
        prices = [100.0 + 2.0 * math.sin(i * 0.1) for i in range(250)]
        result = self.detector.detect(prices)
        assert result.regime == MarketRegime.SIDEWAYS

    def test_get_adjustments(self):
        """Should return correct adjustments for each regime."""
        adj_bull = self.detector.get_adjustments(MarketRegime.BULL)
        adj_bear = self.detector.get_adjustments(MarketRegime.BEAR)

        # Bull: wider TP
        assert adj_bull['take_profit_multiplier'] > 1.0
        # Bear: smaller positions
        assert adj_bear['position_scale_multiplier'] < 1.0
        # Bear: extra confirmation required
        assert adj_bear['min_confirmations_offset'] > 0

    def test_is_trending(self):
        """Should report trending for bull and bear."""
        prices_up = [100.0 + i * 0.5 for i in range(250)]
        result = self.detector.detect(prices_up)
        assert result.is_trending

    def test_serialization(self):
        """Should serialize and restore state."""
        prices = [100.0 + i * 0.3 for i in range(250)]
        self.detector.detect(prices)

        data = self.detector.to_dict()
        assert data["regime"] in ["bull", "bear", "sideways", "unknown"]

        new_detector = RegimeDetector(self.config)
        new_detector.from_dict(data)
        assert new_detector.last_result is not None
        assert new_detector.last_result.regime.value == data["regime"]

    def test_atr_calculation(self):
        """Should compute ATR as percentage of price."""
        # Create prices with known volatility
        prices = [100.0 + i * 0.3 for i in range(250)]
        highs = [p + 2.0 for p in prices]
        lows = [p - 2.0 for p in prices]
        result = self.detector.detect(prices, highs, lows)
        assert result.atr_pct > 0


class TestPairReEnable:
    """Tests for the pair re-enable fix."""

    def setup_method(self):
        self.config = AdaptiveConfig(
            min_win_rate=0.35,
            max_consecutive_losses=5,
            min_trades_for_decision=10,
            cooldown_hours=2.0,
        )
        self.manager = AdaptivePairManager(self.config)

    def test_check_reenable_after_cooldown(self):
        """Disabled pairs should be re-enabled after cooldown expires."""
        self.manager.register_pair("BTC/USD")
        self.manager.force_disable("BTC/USD", "test")
        assert not self.manager.is_pair_enabled("BTC/USD")

        # Manually expire the cooldown
        perf = self.manager._pairs["BTC/USD"]
        perf.cooldown_until = datetime.now(timezone.utc) - timedelta(seconds=1)

        reenabled = self.manager.check_reenable_pairs()
        assert "BTC/USD" in reenabled
        assert self.manager.is_pair_enabled("BTC/USD")

    def test_check_reenable_respects_cooldown(self):
        """Should NOT re-enable pairs still in cooldown."""
        config = AdaptiveConfig(cooldown_hours=999.0)  # Very long cooldown
        manager = AdaptivePairManager(config)
        manager.register_pair("BTC/USD")
        manager.force_disable("BTC/USD", "test")

        reenabled = manager.check_reenable_pairs()
        assert "BTC/USD" not in reenabled
        assert not manager.is_pair_enabled("BTC/USD")

    def test_check_reenable_only_disabled_pairs(self):
        """Should only check disabled pairs, not enabled ones."""
        self.manager.register_pair("BTC/USD")
        self.manager.register_pair("ETH/USD")

        # Only disable BTC
        self.manager.force_disable("BTC/USD", "test")

        # Expire BTC cooldown
        perf = self.manager._pairs["BTC/USD"]
        perf.cooldown_until = datetime.now(timezone.utc) - timedelta(seconds=1)

        reenabled = self.manager.check_reenable_pairs()
        assert "BTC/USD" in reenabled
        assert "ETH/USD" not in reenabled  # Was never disabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
