"""
Stochastic Oscillator indicator.

The Stochastic Oscillator compares a closing price to its price range
over a given period. It generates overbought/oversold signals that
complement RSI for stronger confirmation.

%K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
%D = SMA(%K, smooth_period)
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class StochasticResult:
    """Stochastic Oscillator calculation result."""
    k_value: float  # Fast %K line
    d_value: float  # Slow %D line (signal line)
    is_oversold: bool
    is_overbought: bool
    crossover_signal: str  # "bullish", "bearish", or "neutral"
    signal: str  # "buy", "sell", or "neutral"


class StochasticIndicator:
    """
    Stochastic Oscillator indicator.

    Measures momentum by comparing closing price to the high-low range.
    Works best as confirmation alongside RSI:
    - Both RSI < 30 AND Stochastic %K < 20 = strong oversold signal
    - Both RSI > 70 AND Stochastic %K > 80 = strong overbought signal
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        oversold: float = 20.0,
        overbought: float = 80.0
    ):
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> Optional[StochasticResult]:
        """
        Calculate Stochastic Oscillator.

        Args:
            highs: List of high prices (oldest first).
            lows: List of low prices (oldest first).
            closes: List of close prices (oldest first).

        Returns:
            StochasticResult or None if insufficient data.
        """
        min_data = self.k_period + self.d_period
        if len(closes) < min_data or len(highs) < min_data or len(lows) < min_data:
            return None

        # Calculate raw %K values for the last d_period candles (for %D smoothing)
        k_values = []
        for i in range(len(closes) - self.d_period, len(closes)):
            start = i - self.k_period + 1
            if start < 0:
                continue
            highest_high = max(highs[start:i + 1])
            lowest_low = min(lows[start:i + 1])
            price_range = highest_high - lowest_low
            if price_range > 0:
                k = 100.0 * (closes[i] - lowest_low) / price_range
            else:
                k = 50.0  # Neutral if no range
            k_values.append(k)

        if not k_values:
            return None

        k_value = k_values[-1]
        d_value = sum(k_values) / len(k_values)

        is_oversold = k_value < self.oversold
        is_overbought = k_value > self.overbought

        # Crossover detection: %K crossing %D
        crossover_signal = "neutral"
        if len(k_values) >= 2:
            prev_k = k_values[-2]
            # Calculate previous %D (approximate)
            if len(k_values) >= 3:
                prev_d = sum(k_values[:-1]) / len(k_values[:-1])
            else:
                prev_d = d_value
            if prev_k <= prev_d and k_value > d_value:
                crossover_signal = "bullish"
            elif prev_k >= prev_d and k_value < d_value:
                crossover_signal = "bearish"

        # Signal
        if is_oversold and crossover_signal == "bullish":
            signal = "buy"
        elif is_overbought and crossover_signal == "bearish":
            signal = "sell"
        elif is_oversold:
            signal = "buy"
        elif is_overbought:
            signal = "sell"
        else:
            signal = "neutral"

        return StochasticResult(
            k_value=k_value,
            d_value=d_value,
            is_oversold=is_oversold,
            is_overbought=is_overbought,
            crossover_signal=crossover_signal,
            signal=signal
        )

    def reset(self) -> None:
        """Reset indicator state."""
        pass  # Stateless calculation


def calculate_stochastic(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20.0,
    overbought: float = 80.0
) -> Optional[StochasticResult]:
    """Convenience function to calculate Stochastic Oscillator."""
    indicator = StochasticIndicator(k_period, d_period, oversold, overbought)
    return indicator.calculate(highs, lows, closes)
