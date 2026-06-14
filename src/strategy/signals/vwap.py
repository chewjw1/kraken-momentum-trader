"""
Volume Weighted Average Price (VWAP) indicator.

VWAP is used as a benchmark for intraday trading, showing the average
price weighted by volume. Price above VWAP = bullish bias, below = bearish.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class VWAPResult:
    """VWAP calculation result."""
    value: float
    price_vs_vwap: float  # Percentage above/below VWAP
    is_above: bool
    is_below: bool
    signal: str  # "bullish", "bearish", or "neutral"


class VWAPIndicator:
    """
    Volume Weighted Average Price indicator.

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    Used for:
    - Identifying intraday trend direction
    - Dynamic support/resistance levels
    - Entry timing (buy below VWAP in uptrend, sell above in downtrend)
    """

    def __init__(self, threshold_percent: float = 0.5, anchor_candles: int = 0):
        """
        Initialize VWAP indicator.

        Args:
            threshold_percent: Percentage distance from VWAP to generate signals.
            anchor_candles: If > 0, compute VWAP over only the last N candles
                (a rolling "session" anchor). If 0 (default), VWAP is cumulative
                over the entire input window — which, for a 540-candle lookback
                on 4h candles, is a ~90-day volume-weighted average that behaves
                as a slow trend filter rather than an intraday mean-reversion
                level. Anchoring to a recent window restores VWAP's role as a
                session reference price.
        """
        self.threshold_percent = threshold_percent
        self.anchor_candles = anchor_candles
        self._cumulative_tp_volume: float = 0.0
        self._cumulative_volume: float = 0.0

    def _anchor(self, series: List[float]) -> List[float]:
        """Slice a series to the anchor window (last N candles) if configured."""
        if self.anchor_candles and self.anchor_candles > 0:
            return series[-self.anchor_candles:]
        return series

    def calculate(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> Optional[VWAPResult]:
        """
        Calculate VWAP from OHLCV data.

        Args:
            highs: List of high prices.
            lows: List of low prices.
            closes: List of close prices.
            volumes: List of volumes.

        Returns:
            VWAPResult or None if insufficient data.
        """
        if len(closes) < 2 or len(highs) != len(lows) != len(closes) != len(volumes):
            return None

        current_price = closes[-1]

        # Anchor to a recent session window if configured (slice all series
        # consistently so the volume weighting stays aligned).
        a_highs = self._anchor(highs)
        a_lows = self._anchor(lows)
        a_closes = self._anchor(closes)
        a_volumes = self._anchor(volumes)

        # Calculate typical prices
        typical_prices = [
            (h + l + c) / 3
            for h, l, c in zip(a_highs, a_lows, a_closes)
        ]

        # Calculate cumulative values
        cumulative_tp_volume = sum(tp * v for tp, v in zip(typical_prices, a_volumes))
        cumulative_volume = sum(a_volumes)

        if cumulative_volume == 0:
            return None

        vwap = cumulative_tp_volume / cumulative_volume

        # Calculate percentage distance from VWAP
        price_vs_vwap = ((current_price - vwap) / vwap) * 100

        is_above = price_vs_vwap > self.threshold_percent
        is_below = price_vs_vwap < -self.threshold_percent

        if is_above:
            signal = "bullish"
        elif is_below:
            signal = "bearish"
        else:
            signal = "neutral"

        return VWAPResult(
            value=vwap,
            price_vs_vwap=price_vs_vwap,
            is_above=is_above,
            is_below=is_below,
            signal=signal
        )

    def calculate_with_bands(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        std_dev_multiplier: float = 2.0
    ) -> Optional[dict]:
        """
        Calculate VWAP with standard deviation bands.

        Args:
            highs: List of high prices.
            lows: List of low prices.
            closes: List of close prices.
            volumes: List of volumes.
            std_dev_multiplier: Multiplier for standard deviation bands.

        Returns:
            Dictionary with VWAP, upper_band, lower_band, and VWAPResult.
        """
        result = self.calculate(highs, lows, closes, volumes)
        if result is None:
            return None

        # Use the same anchored window the VWAP value was computed over.
        a_highs = self._anchor(highs)
        a_lows = self._anchor(lows)
        a_closes = self._anchor(closes)
        a_volumes = self._anchor(volumes)

        # Calculate typical prices
        typical_prices = [
            (h + l + c) / 3
            for h, l, c in zip(a_highs, a_lows, a_closes)
        ]

        # Calculate variance for bands
        cumulative_volume = sum(a_volumes)
        if cumulative_volume == 0:
            return None

        # Volume-weighted variance
        variance_sum = sum(
            v * (tp - result.value) ** 2
            for tp, v in zip(typical_prices, a_volumes)
        )
        variance = variance_sum / cumulative_volume
        std_dev = variance ** 0.5

        upper_band = result.value + (std_dev * std_dev_multiplier)
        lower_band = result.value - (std_dev * std_dev_multiplier)

        return {
            "vwap": result.value,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "result": result
        }

    def reset(self) -> None:
        """Reset indicator state for new trading session."""
        self._cumulative_tp_volume = 0.0
        self._cumulative_volume = 0.0


def calculate_vwap(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    threshold_percent: float = 0.5
) -> Optional[VWAPResult]:
    """
    Convenience function to calculate VWAP.

    Args:
        highs: List of high prices.
        lows: List of low prices.
        closes: List of close prices.
        volumes: List of volumes.
        threshold_percent: Signal threshold percentage.

    Returns:
        VWAPResult or None if insufficient data.
    """
    indicator = VWAPIndicator(threshold_percent)
    return indicator.calculate(highs, lows, closes, volumes)
