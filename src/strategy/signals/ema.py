"""
Exponential Moving Average (EMA) indicator with crossover detection.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EMAResult:
    """EMA calculation result."""
    short_ema: float
    long_ema: float
    price: float
    price_above_ema: bool
    crossover_signal: str  # "bullish", "bearish", or "neutral"
    trend_strength: float  # Percentage difference between EMAs


class EMAIndicator:
    """
    Exponential Moving Average indicator with crossover detection.

    EMA gives more weight to recent prices:
    EMA = Price * k + EMA_prev * (1 - k)
    k = 2 / (period + 1)
    """

    def __init__(
        self,
        short_period: int = 12,
        long_period: int = 26
    ):
        """
        Initialize EMA indicator.

        Args:
            short_period: Period for short-term EMA.
            long_period: Period for long-term EMA.
        """
        if short_period >= long_period:
            raise ValueError("Short period must be less than long period")

        self.short_period = short_period
        self.long_period = long_period

        # Smoothing factors
        self.short_k = 2.0 / (short_period + 1)
        self.long_k = 2.0 / (long_period + 1)

        # State
        self._prices: list[float] = []
        self._short_ema: Optional[float] = None
        self._long_ema: Optional[float] = None
        self._prev_short_ema: Optional[float] = None
        self._prev_long_ema: Optional[float] = None

    def calculate(self, prices: list[float]) -> Optional[EMAResult]:
        """
        Calculate EMAs from a list of closing prices.

        Args:
            prices: List of closing prices (oldest first).

        Returns:
            EMAResult or None if insufficient data.
        """
        if len(prices) < self.long_period:
            return None

        # Calculate initial SMAs for seeding
        short_sma = sum(prices[:self.short_period]) / self.short_period
        long_sma = sum(prices[:self.long_period]) / self.long_period

        # Calculate EMAs
        short_ema = short_sma
        long_ema = long_sma

        # Apply EMA calculation from after initial SMA periods
        for i in range(self.short_period, len(prices)):
            short_ema = prices[i] * self.short_k + short_ema * (1 - self.short_k)

        for i in range(self.long_period, len(prices)):
            long_ema = prices[i] * self.long_k + long_ema * (1 - self.long_k)

        current_price = prices[-1]

        return self._create_result(short_ema, long_ema, current_price)

    def update(self, price: float) -> Optional[EMAResult]:
        """
        Update EMAs with a new price (streaming mode).

        Args:
            price: New closing price.

        Returns:
            EMAResult or None if insufficient data.
        """
        self._prices.append(price)

        # Keep only necessary prices
        max_prices = self.long_period * 3
        if len(self._prices) > max_prices:
            self._prices = self._prices[-max_prices:]

        if len(self._prices) < self.long_period:
            return None

        # Save previous values for crossover detection
        self._prev_short_ema = self._short_ema
        self._prev_long_ema = self._long_ema

        # Initialize or update short EMA
        if self._short_ema is None:
            self._short_ema = sum(self._prices[:self.short_period]) / self.short_period
            for p in self._prices[self.short_period:]:
                self._short_ema = p * self.short_k + self._short_ema * (1 - self.short_k)
        else:
            self._short_ema = price * self.short_k + self._short_ema * (1 - self.short_k)

        # Initialize or update long EMA
        if self._long_ema is None:
            self._long_ema = sum(self._prices[:self.long_period]) / self.long_period
            for p in self._prices[self.long_period:]:
                self._long_ema = p * self.long_k + self._long_ema * (1 - self.long_k)
        else:
            self._long_ema = price * self.long_k + self._long_ema * (1 - self.long_k)

        return self._create_result(self._short_ema, self._long_ema, price)

    def _create_result(
        self,
        short_ema: float,
        long_ema: float,
        price: float
    ) -> EMAResult:
        """Create EMAResult with crossover detection."""
        price_above_ema = price > long_ema

        # Calculate trend strength
        trend_strength = (short_ema - long_ema) / long_ema * 100 if long_ema != 0 else 0

        # Detect crossover
        crossover_signal = "neutral"

        if self._prev_short_ema is not None and self._prev_long_ema is not None:
            # Bullish crossover: short EMA crosses above long EMA
            if self._prev_short_ema <= self._prev_long_ema and short_ema > long_ema:
                crossover_signal = "bullish"
            # Bearish crossover: short EMA crosses below long EMA
            elif self._prev_short_ema >= self._prev_long_ema and short_ema < long_ema:
                crossover_signal = "bearish"

        return EMAResult(
            short_ema=short_ema,
            long_ema=long_ema,
            price=price,
            price_above_ema=price_above_ema,
            crossover_signal=crossover_signal,
            trend_strength=trend_strength
        )

    def reset(self) -> None:
        """Reset indicator state."""
        self._prices = []
        self._short_ema = None
        self._long_ema = None
        self._prev_short_ema = None
        self._prev_long_ema = None


def calculate_ema(prices: list[float], period: int) -> Optional[float]:
    """
    Calculate a single EMA value.

    Args:
        prices: List of closing prices.
        period: EMA period.

    Returns:
        EMA value or None if insufficient data.
    """
    if len(prices) < period:
        return None

    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period

    for price in prices[period:]:
        ema = price * k + ema * (1 - k)

    return ema


def calculate_ema_crossover(
    prices: list[float],
    short_period: int = 12,
    long_period: int = 26
) -> Optional[EMAResult]:
    """
    Convenience function to calculate EMA crossover.

    Args:
        prices: List of closing prices.
        short_period: Short EMA period.
        long_period: Long EMA period.

    Returns:
        EMAResult or None if insufficient data.
    """
    indicator = EMAIndicator(short_period, long_period)
    return indicator.calculate(prices)
