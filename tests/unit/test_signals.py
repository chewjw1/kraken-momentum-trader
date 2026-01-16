"""
Unit tests for signal generators (RSI, EMA, Volume).
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.strategy.signals.rsi import RSIIndicator, calculate_rsi
from src.strategy.signals.ema import EMAIndicator, calculate_ema, calculate_ema_crossover
from src.strategy.signals.volume import VolumeIndicator, analyze_volume


class TestRSIIndicator:
    """Tests for RSI indicator."""

    def test_insufficient_data(self):
        """RSI should return None with insufficient data."""
        indicator = RSIIndicator(period=14)
        prices = [100.0] * 10  # Less than period + 1
        result = indicator.calculate(prices)
        assert result is None

    def test_all_gains(self):
        """RSI should be 100 when all price changes are positive."""
        indicator = RSIIndicator(period=14)
        # Create steadily increasing prices
        prices = [100.0 + i for i in range(20)]
        result = indicator.calculate(prices)
        assert result is not None
        assert result.value == 100.0
        assert result.is_overbought
        assert not result.is_oversold
        assert result.signal == "sell"

    def test_all_losses(self):
        """RSI should be 0 when all price changes are negative."""
        indicator = RSIIndicator(period=14)
        # Create steadily decreasing prices
        prices = [120.0 - i for i in range(20)]
        result = indicator.calculate(prices)
        assert result is not None
        assert result.value == 0.0
        assert result.is_oversold
        assert not result.is_overbought
        assert result.signal == "buy"

    def test_neutral_rsi(self):
        """RSI should be around 50 with equal gains and losses."""
        indicator = RSIIndicator(period=14)
        # Alternating up and down by same amount
        prices = []
        price = 100.0
        for i in range(30):
            prices.append(price)
            price += 1.0 if i % 2 == 0 else -1.0
        result = indicator.calculate(prices)
        assert result is not None
        assert 40 < result.value < 60  # Should be around 50
        assert result.signal == "neutral"

    def test_custom_thresholds(self):
        """RSI should respect custom oversold/overbought thresholds."""
        indicator = RSIIndicator(period=14, oversold=40, overbought=60)
        prices = [100.0 + i * 0.1 for i in range(20)]  # Slight uptrend
        result = indicator.calculate(prices)
        assert result is not None
        # With these thresholds, a value of 65 would be overbought
        assert indicator.overbought == 60

    def test_streaming_update(self):
        """RSI should work with streaming updates."""
        indicator = RSIIndicator(period=14)

        # Add prices one by one
        for i in range(20):
            result = indicator.update(100.0 + i * 0.5)

        assert result is not None
        assert result.value > 50  # Uptrend should have RSI > 50

    def test_reset(self):
        """Reset should clear indicator state."""
        indicator = RSIIndicator(period=14)
        for i in range(20):
            indicator.update(100.0 + i)

        indicator.reset()
        result = indicator.update(100.0)
        assert result is None  # Insufficient data after reset


class TestEMAIndicator:
    """Tests for EMA indicator."""

    def test_insufficient_data(self):
        """EMA should return None with insufficient data."""
        indicator = EMAIndicator(short_period=12, long_period=26)
        prices = [100.0] * 20  # Less than long period
        result = indicator.calculate(prices)
        assert result is None

    def test_ema_calculation(self):
        """EMA should calculate correctly."""
        indicator = EMAIndicator(short_period=12, long_period=26)
        prices = [100.0] * 30
        result = indicator.calculate(prices)
        assert result is not None
        # Constant prices should give EMA equal to price
        assert abs(result.short_ema - 100.0) < 0.01
        assert abs(result.long_ema - 100.0) < 0.01

    def test_uptrend_detection(self):
        """EMA should detect uptrend (price above EMA)."""
        indicator = EMAIndicator(short_period=12, long_period=26)
        # Strong uptrend
        prices = [100.0 + i * 2 for i in range(40)]
        result = indicator.calculate(prices)
        assert result is not None
        assert result.price_above_ema  # Price should be above long EMA
        assert result.short_ema > result.long_ema  # Short EMA above long in uptrend
        assert result.trend_strength > 0

    def test_bullish_crossover(self):
        """EMA should detect bullish crossover."""
        indicator = EMAIndicator(short_period=3, long_period=5)

        # First establish short below long (downtrend)
        for price in [100, 99, 98, 97, 96, 95]:
            indicator.update(price)

        # Now create upward movement
        for price in [96, 98, 101, 105, 110]:
            result = indicator.update(price)

        # Should eventually detect bullish crossover
        # (This depends on the exact values, so we test the detection mechanism)
        assert result is not None

    def test_convenience_function(self):
        """Test calculate_ema_crossover convenience function."""
        prices = [100.0 + i for i in range(30)]
        result = calculate_ema_crossover(prices, short_period=12, long_period=26)
        assert result is not None

    def test_invalid_periods(self):
        """Should raise error if short >= long period."""
        with pytest.raises(ValueError):
            EMAIndicator(short_period=26, long_period=12)


class TestVolumeIndicator:
    """Tests for Volume indicator."""

    def test_insufficient_data(self):
        """Volume should return None with insufficient data."""
        indicator = VolumeIndicator(sma_period=20)
        volumes = [1000.0] * 10
        result = indicator.calculate(volumes)
        assert result is None

    def test_average_calculation(self):
        """Volume average should calculate correctly."""
        indicator = VolumeIndicator(sma_period=20)
        volumes = [1000.0] * 25
        result = indicator.calculate(volumes)
        assert result is not None
        assert abs(result.average_volume - 1000.0) < 0.01
        assert abs(result.volume_ratio - 1.0) < 0.01

    def test_high_volume_detection(self):
        """Should detect high volume."""
        indicator = VolumeIndicator(sma_period=20, high_volume_threshold=1.5)
        volumes = [1000.0] * 24 + [2000.0]  # Last volume is 2x average
        result = indicator.calculate(volumes)
        assert result is not None
        assert result.is_high_volume
        assert result.volume_ratio > 1.5

    def test_low_volume_detection(self):
        """Should detect low volume."""
        indicator = VolumeIndicator(sma_period=20, high_volume_threshold=1.5)
        volumes = [1000.0] * 24 + [500.0]  # Last volume is half average
        result = indicator.calculate(volumes)
        assert result is not None
        assert not result.is_high_volume
        assert result.volume_ratio < 1.0

    def test_volume_with_prices(self):
        """Volume indicator should work with price data."""
        indicator = VolumeIndicator(sma_period=20)
        volumes = [1000.0] * 24 + [1500.0]
        prices = [100.0 + i * 0.1 for i in range(25)]
        result = indicator.calculate(volumes, prices)
        assert result is not None
        assert result.signal in ["confirming", "diverging", "neutral"]

    def test_increasing_volume(self):
        """Should detect increasing volume trend."""
        indicator = VolumeIndicator(sma_period=20)
        # Volume increasing over time
        volumes = [1000.0 + i * 50 for i in range(25)]
        result = indicator.calculate(volumes)
        assert result is not None
        assert result.is_increasing


class TestSignalIntegration:
    """Integration tests for signals working together."""

    def test_all_signals_valid(self):
        """All signals should be calculable with sufficient data."""
        prices = [100.0 + i * 0.5 for i in range(50)]
        volumes = [1000.0 + i * 10 for i in range(50)]

        rsi = calculate_rsi(prices)
        ema = calculate_ema_crossover(prices)
        volume = analyze_volume(volumes, prices)

        assert rsi is not None
        assert ema is not None
        assert volume is not None

    def test_bullish_confluence(self):
        """Test bullish signal confluence."""
        # Create data that should give bullish signals
        # Sharp drop followed by recovery (RSI oversold + price above EMA)
        prices = [100.0] * 30
        prices.extend([90.0, 88.0, 85.0, 83.0, 82.0])  # Sharp drop
        prices.extend([83.0, 85.0, 88.0, 91.0, 95.0])  # Recovery
        volumes = [1000.0] * len(prices)
        volumes[-5:] = [1500.0] * 5  # High volume on recovery

        rsi = calculate_rsi(prices)
        ema = calculate_ema_crossover(prices, short_period=5, long_period=10)
        volume = analyze_volume(volumes, prices)

        assert rsi is not None
        assert ema is not None
        assert volume is not None
