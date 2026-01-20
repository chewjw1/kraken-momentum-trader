"""
Feature extraction for ML model.

Converts raw market data into features for model training and prediction.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..strategy.base_strategy import MarketData
from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLFeatures:
    """Features extracted for ML model."""
    # Price features
    rsi: float
    rsi_slope: float  # Change in RSI over last 3 periods
    price_vs_ema_short: float  # % distance from short EMA
    price_vs_ema_long: float  # % distance from long EMA
    ema_crossover: float  # Short EMA vs Long EMA (positive = bullish)

    # Volume features
    volume_ratio: float  # Current volume / average volume
    volume_trend: float  # Volume change trend

    # Price action features
    price_change_1h: float  # % change last hour (last candle)
    price_change_4h: float  # % change last 4 hours
    price_change_24h: float  # % change last 24 hours
    volatility: float  # Standard deviation of returns

    # Candle features
    candle_body_ratio: float  # Body size / total range (0-1)
    upper_wick_ratio: float  # Upper wick / total range
    lower_wick_ratio: float  # Lower wick / total range

    # Time features
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6 (Monday=0)

    # Trend features
    trend_strength: float  # ADX-like measure
    higher_highs: int  # Count of higher highs in last 5 candles
    lower_lows: int  # Count of lower lows in last 5 candles

    # Support/Resistance
    distance_to_high_20: float  # % distance to 20-period high
    distance_to_low_20: float  # % distance to 20-period low

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array([
            self.rsi,
            self.rsi_slope,
            self.price_vs_ema_short,
            self.price_vs_ema_long,
            self.ema_crossover,
            self.volume_ratio,
            self.volume_trend,
            self.price_change_1h,
            self.price_change_4h,
            self.price_change_24h,
            self.volatility,
            self.candle_body_ratio,
            self.upper_wick_ratio,
            self.lower_wick_ratio,
            self.hour_of_day,
            self.day_of_week,
            self.trend_strength,
            self.higher_highs,
            self.lower_lows,
            self.distance_to_high_20,
            self.distance_to_low_20,
        ])

    @staticmethod
    def feature_names() -> list[str]:
        """Get ordered list of feature names."""
        return [
            "rsi",
            "rsi_slope",
            "price_vs_ema_short",
            "price_vs_ema_long",
            "ema_crossover",
            "volume_ratio",
            "volume_trend",
            "price_change_1h",
            "price_change_4h",
            "price_change_24h",
            "volatility",
            "candle_body_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "hour_of_day",
            "day_of_week",
            "trend_strength",
            "higher_highs",
            "lower_lows",
            "distance_to_high_20",
            "distance_to_low_20",
        ]


class FeatureExtractor:
    """
    Extracts ML features from market data.

    Uses OHLCV data to compute technical indicators and
    derived features for model input.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        ema_short: int = 12,
        ema_long: int = 26,
        volume_period: int = 20
    ):
        """
        Initialize feature extractor.

        Args:
            rsi_period: Period for RSI calculation.
            ema_short: Short EMA period.
            ema_long: Long EMA period.
            volume_period: Period for volume average.
        """
        self.rsi_period = rsi_period
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.volume_period = volume_period

    def extract(self, market_data: MarketData) -> Optional[MLFeatures]:
        """
        Extract features from market data.

        Args:
            market_data: Market data with OHLCV.

        Returns:
            MLFeatures or None if insufficient data.
        """
        try:
            closes = np.array(market_data.closes)
            highs = np.array(market_data.highs)
            lows = np.array(market_data.lows)
            opens = np.array(market_data.opens)
            volumes = np.array(market_data.volumes)

            if len(closes) < max(self.rsi_period, self.ema_long, self.volume_period, 24) + 5:
                logger.warning(f"Insufficient data for feature extraction: {len(closes)} candles")
                return None

            # Current price
            current_price = closes[-1]

            # RSI
            rsi = self._calculate_rsi(closes)
            rsi_prev = self._calculate_rsi(closes[:-3])
            rsi_slope = rsi - rsi_prev if rsi_prev else 0

            # EMAs
            ema_short = self._calculate_ema(closes, self.ema_short)
            ema_long = self._calculate_ema(closes, self.ema_long)
            price_vs_ema_short = ((current_price - ema_short) / ema_short) * 100
            price_vs_ema_long = ((current_price - ema_long) / ema_long) * 100
            ema_crossover = ((ema_short - ema_long) / ema_long) * 100

            # Volume
            avg_volume = np.mean(volumes[-self.volume_period:])
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            volume_trend = (np.mean(volumes[-5:]) / np.mean(volumes[-20:-5])) if np.mean(volumes[-20:-5]) > 0 else 1.0

            # Price changes
            price_change_1h = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) >= 2 else 0
            price_change_4h = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 else 0
            price_change_24h = ((closes[-1] - closes[-25]) / closes[-25]) * 100 if len(closes) >= 25 else 0

            # Volatility (standard deviation of returns)
            returns = np.diff(closes[-20:]) / closes[-20:-1]
            volatility = np.std(returns) * 100 if len(returns) > 0 else 0

            # Candle features (last candle)
            candle_range = highs[-1] - lows[-1]
            if candle_range > 0:
                body = abs(closes[-1] - opens[-1])
                candle_body_ratio = body / candle_range
                upper_wick = highs[-1] - max(closes[-1], opens[-1])
                lower_wick = min(closes[-1], opens[-1]) - lows[-1]
                upper_wick_ratio = upper_wick / candle_range
                lower_wick_ratio = lower_wick / candle_range
            else:
                candle_body_ratio = 0.5
                upper_wick_ratio = 0.25
                lower_wick_ratio = 0.25

            # Time features
            now = datetime.now(timezone.utc)
            hour_of_day = now.hour
            day_of_week = now.weekday()

            # Trend strength (simplified ADX-like)
            trend_strength = abs(price_change_24h) / (volatility + 0.01)

            # Higher highs / Lower lows
            higher_highs = sum(1 for i in range(-5, -1) if highs[i] > highs[i-1])
            lower_lows = sum(1 for i in range(-5, -1) if lows[i] < lows[i-1])

            # Support/Resistance
            high_20 = np.max(highs[-20:])
            low_20 = np.min(lows[-20:])
            distance_to_high_20 = ((high_20 - current_price) / current_price) * 100
            distance_to_low_20 = ((current_price - low_20) / current_price) * 100

            return MLFeatures(
                rsi=rsi,
                rsi_slope=rsi_slope,
                price_vs_ema_short=price_vs_ema_short,
                price_vs_ema_long=price_vs_ema_long,
                ema_crossover=ema_crossover,
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                price_change_1h=price_change_1h,
                price_change_4h=price_change_4h,
                price_change_24h=price_change_24h,
                volatility=volatility,
                candle_body_ratio=candle_body_ratio,
                upper_wick_ratio=upper_wick_ratio,
                lower_wick_ratio=lower_wick_ratio,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                trend_strength=trend_strength,
                higher_highs=higher_highs,
                lower_lows=lower_lows,
                distance_to_high_20=distance_to_high_20,
                distance_to_low_20=distance_to_low_20,
            )

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def extract_from_ohlcv(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        timestamp: Optional[datetime] = None
    ) -> Optional[MLFeatures]:
        """
        Extract features from raw OHLCV arrays.

        Useful for backtesting/training where MarketData isn't available.

        Args:
            opens: Open prices.
            highs: High prices.
            lows: Low prices.
            closes: Close prices.
            volumes: Volumes.
            timestamp: Optional timestamp for time features.

        Returns:
            MLFeatures or None if insufficient data.
        """
        try:
            closes_arr = np.array(closes)
            highs_arr = np.array(highs)
            lows_arr = np.array(lows)
            opens_arr = np.array(opens)
            volumes_arr = np.array(volumes)

            min_required = max(self.rsi_period, self.ema_long, self.volume_period, 24) + 5
            if len(closes_arr) < min_required:
                return None

            current_price = closes_arr[-1]

            # RSI
            rsi = self._calculate_rsi(closes_arr)
            rsi_prev = self._calculate_rsi(closes_arr[:-3])
            rsi_slope = rsi - rsi_prev if rsi_prev else 0

            # EMAs
            ema_short = self._calculate_ema(closes_arr, self.ema_short)
            ema_long = self._calculate_ema(closes_arr, self.ema_long)
            price_vs_ema_short = ((current_price - ema_short) / ema_short) * 100
            price_vs_ema_long = ((current_price - ema_long) / ema_long) * 100
            ema_crossover = ((ema_short - ema_long) / ema_long) * 100

            # Volume
            avg_volume = np.mean(volumes_arr[-self.volume_period:])
            volume_ratio = volumes_arr[-1] / avg_volume if avg_volume > 0 else 1.0
            volume_trend = (np.mean(volumes_arr[-5:]) / np.mean(volumes_arr[-20:-5])) if np.mean(volumes_arr[-20:-5]) > 0 else 1.0

            # Price changes
            price_change_1h = ((closes_arr[-1] - closes_arr[-2]) / closes_arr[-2]) * 100
            price_change_4h = ((closes_arr[-1] - closes_arr[-5]) / closes_arr[-5]) * 100 if len(closes_arr) >= 5 else 0
            price_change_24h = ((closes_arr[-1] - closes_arr[-25]) / closes_arr[-25]) * 100 if len(closes_arr) >= 25 else 0

            # Volatility
            returns = np.diff(closes_arr[-20:]) / closes_arr[-20:-1]
            volatility = np.std(returns) * 100 if len(returns) > 0 else 0

            # Candle features
            candle_range = highs_arr[-1] - lows_arr[-1]
            if candle_range > 0:
                body = abs(closes_arr[-1] - opens_arr[-1])
                candle_body_ratio = body / candle_range
                upper_wick = highs_arr[-1] - max(closes_arr[-1], opens_arr[-1])
                lower_wick = min(closes_arr[-1], opens_arr[-1]) - lows_arr[-1]
                upper_wick_ratio = upper_wick / candle_range
                lower_wick_ratio = lower_wick / candle_range
            else:
                candle_body_ratio = 0.5
                upper_wick_ratio = 0.25
                lower_wick_ratio = 0.25

            # Time features
            ts = timestamp or datetime.now(timezone.utc)
            hour_of_day = ts.hour
            day_of_week = ts.weekday()

            # Trend
            trend_strength = abs(price_change_24h) / (volatility + 0.01)
            higher_highs = sum(1 for i in range(-5, -1) if highs_arr[i] > highs_arr[i-1])
            lower_lows = sum(1 for i in range(-5, -1) if lows_arr[i] < lows_arr[i-1])

            # Support/Resistance
            high_20 = np.max(highs_arr[-20:])
            low_20 = np.min(lows_arr[-20:])
            distance_to_high_20 = ((high_20 - current_price) / current_price) * 100
            distance_to_low_20 = ((current_price - low_20) / current_price) * 100

            return MLFeatures(
                rsi=rsi,
                rsi_slope=rsi_slope,
                price_vs_ema_short=price_vs_ema_short,
                price_vs_ema_long=price_vs_ema_long,
                ema_crossover=ema_crossover,
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                price_change_1h=price_change_1h,
                price_change_4h=price_change_4h,
                price_change_24h=price_change_24h,
                volatility=volatility,
                candle_body_ratio=candle_body_ratio,
                upper_wick_ratio=upper_wick_ratio,
                lower_wick_ratio=lower_wick_ratio,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                trend_strength=trend_strength,
                higher_highs=higher_highs,
                lower_lows=lower_lows,
                distance_to_high_20=distance_to_high_20,
                distance_to_low_20=distance_to_low_20,
            )

        except Exception as e:
            logger.error(f"Feature extraction from OHLCV failed: {e}")
            return None

    def _calculate_rsi(self, closes: np.ndarray) -> float:
        """Calculate RSI."""
        if len(closes) < self.rsi_period + 1:
            return 50.0

        deltas = np.diff(closes[-(self.rsi_period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0

        multiplier = 2 / (period + 1)
        ema = data[-period]

        for price in data[-period + 1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema
