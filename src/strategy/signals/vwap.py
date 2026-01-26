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

    def __init__(self, threshold_percent: float = 0.5):
        """
        Initialize VWAP indicator.

        Args:
            threshold_percent: Percentage distance from VWAP to generate signals.
        """
        self.threshold_percent = threshold_percent
        self._cumulative_tp_volume: float = 0.0
        self._cumulative_volume: float = 0.0

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

        # Calculate typical prices
        typical_prices = [
            (h + l + c) / 3
            for h, l, c in zip(highs, lows, closes)
        ]

        # Calculate cumulative values
        cumulative_tp_volume = sum(tp * v for tp, v in zip(typical_prices, volumes))
        cumulative_volume = sum(volumes)

        if cumulative_volume == 0:
            return None

        vwap = cumulative_tp_volume / cumulative_volume
        current_price = closes[-1]

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

        # Calculate typical prices
        typical_prices = [
            (h + l + c) / 3
            for h, l, c in zip(highs, lows, closes)
        ]

        # Calculate variance for bands
        cumulative_volume = sum(volumes)
        if cumulative_volume == 0:
            return None

        # Volume-weighted variance
        variance_sum = sum(
            v * (tp - result.value) ** 2
            for tp, v in zip(typical_prices, volumes)
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
