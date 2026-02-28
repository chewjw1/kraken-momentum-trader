"""
Moving Average Convergence Divergence (MACD) indicator.

MACD measures the relationship between two EMAs of price.
Standard configuration: MACD = EMA(12) - EMA(26), Signal = EMA(9) of MACD.

Key signals:
- MACD histogram turning positive while RSI oversold = high-probability long
- MACD histogram turning negative while RSI overbought = high-probability short
- MACD/Signal crossovers confirm momentum direction
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MACDResult:
    """MACD calculation result."""
    macd_line: float      # MACD line (fast EMA - slow EMA)
    signal_line: float    # Signal line (EMA of MACD)
    histogram: float      # MACD - Signal (momentum strength)
    histogram_increasing: bool  # Histogram growing (momentum accelerating)
    crossover_signal: str  # "bullish", "bearish", or "neutral"
    signal: str            # "buy", "sell", or "neutral"


class MACDIndicator:
    """
    MACD indicator for momentum confirmation.

    MACD Line = EMA(fast_period) - EMA(slow_period)
    Signal Line = EMA(signal_period) of MACD Line
    Histogram = MACD Line - Signal Line
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def _ema(self, values: List[float], period: int) -> List[float]:
        """Calculate EMA series from values."""
        if len(values) < period:
            return []

        k = 2.0 / (period + 1)
        ema_values = []
        ema = sum(values[:period]) / period

        for i in range(period - 1):
            ema_values.append(ema)  # Backfill with initial SMA

        ema_values.append(ema)

        for i in range(period, len(values)):
            ema = values[i] * k + ema * (1 - k)
            ema_values.append(ema)

        return ema_values

    def calculate(self, prices: List[float]) -> Optional[MACDResult]:
        """
        Calculate MACD from closing prices.

        Args:
            prices: List of closing prices (oldest first).

        Returns:
            MACDResult or None if insufficient data.
        """
        min_data = self.slow_period + self.signal_period
        if len(prices) < min_data:
            return None

        # Calculate fast and slow EMAs
        fast_ema = self._ema(prices, self.fast_period)
        slow_ema = self._ema(prices, self.slow_period)

        if not fast_ema or not slow_ema:
            return None

        # Calculate MACD line (aligned to slow EMA length)
        macd_values = []
        offset = len(fast_ema) - len(slow_ema)
        for i in range(len(slow_ema)):
            macd_values.append(fast_ema[i + offset] - slow_ema[i])

        if len(macd_values) < self.signal_period:
            return None

        # Calculate signal line (EMA of MACD)
        signal_ema = self._ema(macd_values, self.signal_period)

        if not signal_ema:
            return None

        # Current values
        macd_line = macd_values[-1]
        signal_line = signal_ema[-1]
        histogram = macd_line - signal_line

        # Previous histogram for direction
        prev_histogram = 0.0
        if len(macd_values) >= 2 and len(signal_ema) >= 2:
            prev_histogram = macd_values[-2] - signal_ema[-2]

        histogram_increasing = histogram > prev_histogram

        # Crossover detection
        crossover_signal = "neutral"
        if len(macd_values) >= 2 and len(signal_ema) >= 2:
            prev_macd = macd_values[-2]
            prev_signal = signal_ema[-2]
            if prev_macd <= prev_signal and macd_line > signal_line:
                crossover_signal = "bullish"
            elif prev_macd >= prev_signal and macd_line < signal_line:
                crossover_signal = "bearish"

        # Generate signal
        if crossover_signal == "bullish" and histogram > 0:
            signal = "buy"
        elif crossover_signal == "bearish" and histogram < 0:
            signal = "sell"
        elif histogram > 0 and histogram_increasing:
            signal = "buy"
        elif histogram < 0 and not histogram_increasing:
            signal = "sell"
        else:
            signal = "neutral"

        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            histogram_increasing=histogram_increasing,
            crossover_signal=crossover_signal,
            signal=signal
        )

    def reset(self) -> None:
        """Reset indicator state."""
        pass  # Stateless calculation


def calculate_macd(
    prices: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Optional[MACDResult]:
    """Convenience function to calculate MACD."""
    indicator = MACDIndicator(fast_period, slow_period, signal_period)
    return indicator.calculate(prices)
