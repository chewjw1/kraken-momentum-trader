"""
Relative Strength Index (RSI) indicator.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RSIResult:
    """RSI calculation result."""
    value: float
    is_oversold: bool
    is_overbought: bool
    signal: str  # "buy", "sell", or "neutral"


class RSIIndicator:
    """
    Relative Strength Index indicator.

    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0
    ):
        """
        Initialize RSI indicator.

        Args:
            period: RSI calculation period.
            oversold: Level below which asset is considered oversold.
            overbought: Level above which asset is considered overbought.
        """
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

        # State for streaming calculations
        self._prices: list[float] = []
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None

    def calculate(self, prices: list[float]) -> Optional[RSIResult]:
        """
        Calculate RSI from a list of closing prices.

        Args:
            prices: List of closing prices (oldest first).

        Returns:
            RSIResult or None if insufficient data.
        """
        if len(prices) < self.period + 1:
            return None

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Separate gains and losses
        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]

        # Calculate initial averages using SMA
        avg_gain = sum(gains[:self.period]) / self.period
        avg_loss = sum(losses[:self.period]) / self.period

        # Apply smoothing (Wilder's smoothing method)
        for i in range(self.period, len(gains)):
            avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
            avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period

        # Calculate RSI
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Determine signal
        is_oversold = rsi < self.oversold
        is_overbought = rsi > self.overbought

        if is_oversold:
            signal = "buy"
        elif is_overbought:
            signal = "sell"
        else:
            signal = "neutral"

        return RSIResult(
            value=rsi,
            is_oversold=is_oversold,
            is_overbought=is_overbought,
            signal=signal
        )

    def update(self, price: float) -> Optional[RSIResult]:
        """
        Update RSI with a new price (streaming mode).

        Args:
            price: New closing price.

        Returns:
            RSIResult or None if insufficient data.
        """
        self._prices.append(price)

        # Keep only necessary prices
        max_prices = self.period * 3
        if len(self._prices) > max_prices:
            self._prices = self._prices[-max_prices:]

        return self.calculate(self._prices)

    def reset(self) -> None:
        """Reset indicator state."""
        self._prices = []
        self._avg_gain = None
        self._avg_loss = None


def calculate_rsi(
    prices: list[float],
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0
) -> Optional[RSIResult]:
    """
    Convenience function to calculate RSI.

    Args:
        prices: List of closing prices.
        period: RSI period.
        oversold: Oversold threshold.
        overbought: Overbought threshold.

    Returns:
        RSIResult or None if insufficient data.
    """
    indicator = RSIIndicator(period, oversold, overbought)
    return indicator.calculate(prices)
