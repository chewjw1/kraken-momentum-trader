"""
Average True Range (ATR) indicator.

ATR measures market volatility by decomposing the entire range of an asset
price for a given period. Used for dynamic stop-loss and take-profit sizing.

True Range = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
ATR = SMA(True Range, period)
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ATRResult:
    """ATR calculation result."""
    atr: float               # ATR in absolute price terms
    atr_percent: float       # ATR as percentage of current price
    is_high_volatility: bool  # Above threshold
    is_low_volatility: bool   # Below threshold (squeeze potential)
    suggested_stop_loss: float   # 1.5x ATR as stop loss price (below entry)
    suggested_take_profit: float  # 2x ATR as take profit price (above entry)


class ATRIndicator:
    """
    Average True Range indicator for dynamic risk management.

    Key usage:
    - Dynamic stop loss = entry_price - (atr_stop_multiplier * ATR)
    - Dynamic take profit = entry_price + (atr_tp_multiplier * ATR)
    - In low-volatility: tighter stops, less capital at risk
    - In high-volatility: wider stops, avoid noise stop-outs
    """

    def __init__(
        self,
        period: int = 14,
        high_vol_threshold: float = 2.0,  # ATR% above this = high volatility
        low_vol_threshold: float = 0.5,    # ATR% below this = low volatility
        stop_multiplier: float = 1.5,
        tp_multiplier: float = 2.0
    ):
        self.period = period
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.stop_multiplier = stop_multiplier
        self.tp_multiplier = tp_multiplier

    def calculate(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> Optional[ATRResult]:
        """
        Calculate ATR.

        Args:
            highs: List of high prices (oldest first).
            lows: List of low prices (oldest first).
            closes: List of close prices (oldest first).

        Returns:
            ATRResult or None if insufficient data.
        """
        if len(highs) < self.period + 1 or len(lows) < self.period + 1 or len(closes) < self.period + 1:
            return None

        # Calculate True Range for each period
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)

        if len(true_ranges) < self.period:
            return None

        # ATR = SMA of True Range over period
        atr = sum(true_ranges[-self.period:]) / self.period

        current_price = closes[-1]
        atr_percent = (atr / current_price) * 100 if current_price > 0 else 0.0

        is_high_volatility = atr_percent > self.high_vol_threshold
        is_low_volatility = atr_percent < self.low_vol_threshold

        # Dynamic levels
        suggested_stop_loss = current_price - (self.stop_multiplier * atr)
        suggested_take_profit = current_price + (self.tp_multiplier * atr)

        return ATRResult(
            atr=atr,
            atr_percent=atr_percent,
            is_high_volatility=is_high_volatility,
            is_low_volatility=is_low_volatility,
            suggested_stop_loss=suggested_stop_loss,
            suggested_take_profit=suggested_take_profit
        )

    def get_dynamic_stop_percent(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        min_stop: float = 0.5,
        max_stop: float = 5.0
    ) -> float:
        """
        Calculate dynamic stop loss percentage based on ATR.

        Returns the stop percentage clamped to [min_stop, max_stop].
        """
        result = self.calculate(highs, lows, closes)
        if result is None:
            return min_stop

        stop_pct = result.atr_percent * self.stop_multiplier
        return max(min_stop, min(stop_pct, max_stop))

    def get_dynamic_tp_percent(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        min_tp: float = 1.0,
        max_tp: float = 10.0
    ) -> float:
        """
        Calculate dynamic take profit percentage based on ATR.

        Returns the TP percentage clamped to [min_tp, max_tp].
        """
        result = self.calculate(highs, lows, closes)
        if result is None:
            return min_tp

        tp_pct = result.atr_percent * self.tp_multiplier
        return max(min_tp, min(tp_pct, max_tp))

    def reset(self) -> None:
        """Reset indicator state."""
        pass  # Stateless calculation


def calculate_atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14
) -> Optional[ATRResult]:
    """Convenience function to calculate ATR."""
    indicator = ATRIndicator(period=period)
    return indicator.calculate(highs, lows, closes)
