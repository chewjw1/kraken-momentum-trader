"""
Bollinger Bands indicator.

Bollinger Bands consist of a middle band (SMA) and upper/lower bands
at standard deviations from the middle. Used for volatility and
mean-reversion signals.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BollingerResult:
    """Bollinger Bands calculation result."""
    middle_band: float  # SMA
    upper_band: float
    lower_band: float
    bandwidth: float  # (Upper - Lower) / Middle * 100
    percent_b: float  # (Price - Lower) / (Upper - Lower)
    is_squeeze: bool  # Low volatility (potential breakout)
    is_above_upper: bool
    is_below_lower: bool
    signal: str  # "buy", "sell", or "neutral"


class BollingerBandsIndicator:
    """
    Bollinger Bands indicator.

    Middle Band = SMA(period)
    Upper Band = Middle + (std_dev_multiplier * StdDev)
    Lower Band = Middle - (std_dev_multiplier * StdDev)

    Signals:
    - Price touching lower band + RSI oversold = potential buy
    - Price touching upper band + RSI overbought = potential sell
    - Squeeze (narrow bands) = potential breakout coming
    """

    def __init__(
        self,
        period: int = 20,
        std_dev_multiplier: float = 2.0,
        squeeze_threshold: float = 4.0  # Bandwidth % below this = squeeze
    ):
        """
        Initialize Bollinger Bands indicator.

        Args:
            period: SMA period for middle band.
            std_dev_multiplier: Standard deviation multiplier for bands.
            squeeze_threshold: Bandwidth percentage below which is a squeeze.
        """
        self.period = period
        self.std_dev_multiplier = std_dev_multiplier
        self.squeeze_threshold = squeeze_threshold

    def calculate(self, prices: List[float]) -> Optional[BollingerResult]:
        """
        Calculate Bollinger Bands from closing prices.

        Args:
            prices: List of closing prices (oldest first).

        Returns:
            BollingerResult or None if insufficient data.
        """
        if len(prices) < self.period:
            return None

        # Use last 'period' prices
        recent_prices = prices[-self.period:]
        current_price = prices[-1]

        # Calculate SMA (middle band)
        middle_band = sum(recent_prices) / self.period

        # Calculate standard deviation
        variance = sum((p - middle_band) ** 2 for p in recent_prices) / self.period
        std_dev = variance ** 0.5

        # Calculate bands
        upper_band = middle_band + (self.std_dev_multiplier * std_dev)
        lower_band = middle_band - (self.std_dev_multiplier * std_dev)

        # Calculate bandwidth (volatility measure)
        bandwidth = ((upper_band - lower_band) / middle_band) * 100 if middle_band > 0 else 0

        # Calculate %B (where price is within bands)
        band_range = upper_band - lower_band
        if band_range > 0:
            percent_b = (current_price - lower_band) / band_range
        else:
            percent_b = 0.5

        # Detect squeeze (low volatility)
        is_squeeze = bandwidth < self.squeeze_threshold

        # Detect band touches
        is_above_upper = current_price >= upper_band
        is_below_lower = current_price <= lower_band

        # Generate signal
        if is_below_lower:
            signal = "buy"  # Mean reversion opportunity
        elif is_above_upper:
            signal = "sell"  # Mean reversion opportunity
        else:
            signal = "neutral"

        return BollingerResult(
            middle_band=middle_band,
            upper_band=upper_band,
            lower_band=lower_band,
            bandwidth=bandwidth,
            percent_b=percent_b,
            is_squeeze=is_squeeze,
            is_above_upper=is_above_upper,
            is_below_lower=is_below_lower,
            signal=signal
        )

    def calculate_with_trend(
        self,
        prices: List[float],
        lookback: int = 5
    ) -> Optional[dict]:
        """
        Calculate Bollinger Bands with trend information.

        Args:
            prices: List of closing prices.
            lookback: Number of periods to check for band trend.

        Returns:
            Dictionary with BollingerResult and trend info.
        """
        if len(prices) < self.period + lookback:
            return None

        current_result = self.calculate(prices)
        if current_result is None:
            return None

        # Calculate previous bandwidth for trend
        prev_result = self.calculate(prices[:-lookback])
        if prev_result is None:
            return None

        bandwidth_expanding = current_result.bandwidth > prev_result.bandwidth
        bandwidth_contracting = current_result.bandwidth < prev_result.bandwidth

        return {
            "result": current_result,
            "bandwidth_expanding": bandwidth_expanding,
            "bandwidth_contracting": bandwidth_contracting,
            "bandwidth_change": current_result.bandwidth - prev_result.bandwidth
        }

    def reset(self) -> None:
        """Reset indicator state."""
        pass  # Stateless calculation


def calculate_bollinger(
    prices: List[float],
    period: int = 20,
    std_dev_multiplier: float = 2.0,
    squeeze_threshold: float = 4.0
) -> Optional[BollingerResult]:
    """
    Convenience function to calculate Bollinger Bands.

    Args:
        prices: List of closing prices.
        period: SMA period.
        std_dev_multiplier: Standard deviation multiplier.
        squeeze_threshold: Squeeze detection threshold.

    Returns:
        BollingerResult or None if insufficient data.
    """
    indicator = BollingerBandsIndicator(period, std_dev_multiplier, squeeze_threshold)
    return indicator.calculate(prices)
