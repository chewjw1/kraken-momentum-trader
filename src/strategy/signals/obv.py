"""
On-Balance Volume (OBV) indicator.

OBV tracks cumulative buying/selling pressure using volume flow.
Key signal: OBV divergence from price is one of the strongest
mean-reversion indicators.

- Price falling but OBV rising = hidden bullish divergence (buy signal)
- Price rising but OBV falling = hidden bearish divergence (sell signal)
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class OBVResult:
    """OBV calculation result."""
    obv: float                    # Current OBV value
    obv_sma: float                # SMA of OBV for trend
    obv_trend: str                # "rising", "falling", or "flat"
    divergence: str               # "bullish_divergence", "bearish_divergence", or "none"
    signal: str                   # "buy", "sell", or "neutral"


class OBVIndicator:
    """
    On-Balance Volume indicator.

    OBV adds volume on up-close days and subtracts on down-close days.
    Divergences between OBV and price are strong reversal signals.
    """

    def __init__(
        self,
        sma_period: int = 20,
        divergence_lookback: int = 10
    ):
        self.sma_period = sma_period
        self.divergence_lookback = divergence_lookback

    def calculate(
        self,
        closes: List[float],
        volumes: List[float]
    ) -> Optional[OBVResult]:
        """
        Calculate OBV.

        Args:
            closes: List of closing prices (oldest first).
            volumes: List of volumes (oldest first).

        Returns:
            OBVResult or None if insufficient data.
        """
        min_data = max(self.sma_period, self.divergence_lookback) + 2
        if len(closes) < min_data or len(volumes) < min_data:
            return None

        # Calculate OBV series
        obv_series = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv_series.append(obv_series[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv_series.append(obv_series[-1] - volumes[i])
            else:
                obv_series.append(obv_series[-1])

        current_obv = obv_series[-1]

        # OBV SMA
        obv_sma = sum(obv_series[-self.sma_period:]) / self.sma_period

        # OBV trend (compare recent SMA to older SMA)
        half = self.sma_period // 2
        if len(obv_series) >= self.sma_period + half:
            recent_obv_avg = sum(obv_series[-half:]) / half
            older_obv_avg = sum(obv_series[-(self.sma_period):-half]) / (self.sma_period - half)
            if recent_obv_avg > older_obv_avg * 1.01:
                obv_trend = "rising"
            elif recent_obv_avg < older_obv_avg * 0.99:
                obv_trend = "falling"
            else:
                obv_trend = "flat"
        else:
            obv_trend = "flat"

        # Divergence detection
        divergence = self._detect_divergence(closes, obv_series)

        # Signal
        if divergence == "bullish_divergence":
            signal = "buy"
        elif divergence == "bearish_divergence":
            signal = "sell"
        elif obv_trend == "rising" and current_obv > obv_sma:
            signal = "buy"
        elif obv_trend == "falling" and current_obv < obv_sma:
            signal = "sell"
        else:
            signal = "neutral"

        return OBVResult(
            obv=current_obv,
            obv_sma=obv_sma,
            obv_trend=obv_trend,
            divergence=divergence,
            signal=signal
        )

    def _detect_divergence(
        self,
        closes: List[float],
        obv_series: List[float]
    ) -> str:
        """
        Detect divergence between price and OBV.

        Bullish divergence: price making lower lows but OBV making higher lows
        Bearish divergence: price making higher highs but OBV making lower highs
        """
        lb = self.divergence_lookback
        if len(closes) < lb + 1 or len(obv_series) < lb + 1:
            return "none"

        recent_closes = closes[-lb:]
        older_closes = closes[-(lb * 2):-lb] if len(closes) >= lb * 2 else closes[:lb]
        recent_obv = obv_series[-lb:]
        older_obv = obv_series[-(lb * 2):-lb] if len(obv_series) >= lb * 2 else obv_series[:lb]

        if not older_closes or not older_obv:
            return "none"

        # Compare lows (for bullish divergence)
        recent_price_low = min(recent_closes)
        older_price_low = min(older_closes)
        recent_obv_low = min(recent_obv)
        older_obv_low = min(older_obv)

        # Price making lower low but OBV making higher low
        if recent_price_low < older_price_low and recent_obv_low > older_obv_low:
            return "bullish_divergence"

        # Compare highs (for bearish divergence)
        recent_price_high = max(recent_closes)
        older_price_high = max(older_closes)
        recent_obv_high = max(recent_obv)
        older_obv_high = max(older_obv)

        # Price making higher high but OBV making lower high
        if recent_price_high > older_price_high and recent_obv_high < older_obv_high:
            return "bearish_divergence"

        return "none"

    def reset(self) -> None:
        """Reset indicator state."""
        pass  # Stateless calculation


def calculate_obv(
    closes: List[float],
    volumes: List[float],
    sma_period: int = 20,
    divergence_lookback: int = 10
) -> Optional[OBVResult]:
    """Convenience function to calculate OBV."""
    indicator = OBVIndicator(sma_period, divergence_lookback)
    return indicator.calculate(closes, volumes)
