"""
Volume-based indicators for momentum confirmation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VolumeResult:
    """Volume analysis result."""
    current_volume: float
    average_volume: float
    volume_ratio: float  # current / average
    is_high_volume: bool
    is_increasing: bool
    signal: str  # "confirming", "diverging", or "neutral"


class VolumeIndicator:
    """
    Volume indicator for momentum confirmation.

    Analyzes volume patterns to confirm or deny price movements:
    - High volume on price increases = strong bullish signal
    - High volume on price decreases = strong bearish signal
    - Low volume on any move = weak/unreliable signal
    """

    def __init__(
        self,
        sma_period: int = 20,
        high_volume_threshold: float = 1.5
    ):
        """
        Initialize volume indicator.

        Args:
            sma_period: Period for volume SMA calculation.
            high_volume_threshold: Multiple of average for "high" volume.
        """
        self.sma_period = sma_period
        self.high_volume_threshold = high_volume_threshold

        # State
        self._volumes: list[float] = []
        self._prices: list[float] = []

    def calculate(
        self,
        volumes: list[float],
        prices: Optional[list[float]] = None
    ) -> Optional[VolumeResult]:
        """
        Calculate volume indicators.

        Args:
            volumes: List of volume values (oldest first).
            prices: Optional list of prices for divergence detection.

        Returns:
            VolumeResult or None if insufficient data.
        """
        if len(volumes) < self.sma_period:
            return None

        current_volume = volumes[-1]
        average_volume = sum(volumes[-self.sma_period:]) / self.sma_period

        volume_ratio = current_volume / average_volume if average_volume > 0 else 0
        is_high_volume = volume_ratio >= self.high_volume_threshold

        # Check if volume is increasing
        recent_avg = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else current_volume
        previous_avg = sum(volumes[-10:-5]) / 5 if len(volumes) >= 10 else average_volume
        is_increasing = recent_avg > previous_avg

        # Determine signal based on price movement if available
        signal = "neutral"
        if prices and len(prices) >= 2:
            price_change = prices[-1] - prices[-2]

            if is_high_volume:
                # High volume confirms the price direction
                signal = "confirming"
            elif is_increasing:
                # Increasing volume is mildly confirming
                signal = "confirming" if abs(price_change) > 0 else "neutral"
            else:
                # Low/decreasing volume may indicate weakness
                signal = "diverging" if abs(price_change) > prices[-2] * 0.01 else "neutral"

        return VolumeResult(
            current_volume=current_volume,
            average_volume=average_volume,
            volume_ratio=volume_ratio,
            is_high_volume=is_high_volume,
            is_increasing=is_increasing,
            signal=signal
        )

    def update(self, volume: float, price: Optional[float] = None) -> Optional[VolumeResult]:
        """
        Update indicator with new data (streaming mode).

        Args:
            volume: New volume value.
            price: Optional new price value.

        Returns:
            VolumeResult or None if insufficient data.
        """
        self._volumes.append(volume)
        if price is not None:
            self._prices.append(price)

        # Keep only necessary data
        max_data = self.sma_period * 3
        if len(self._volumes) > max_data:
            self._volumes = self._volumes[-max_data:]
        if len(self._prices) > max_data:
            self._prices = self._prices[-max_data:]

        return self.calculate(
            self._volumes,
            self._prices if self._prices else None
        )

    def reset(self) -> None:
        """Reset indicator state."""
        self._volumes = []
        self._prices = []
