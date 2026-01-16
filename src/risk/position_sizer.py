"""
Position sizing calculator.
Determines appropriate position sizes based on risk parameters.
"""

from dataclasses import dataclass
from typing import Optional

from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSize:
    """Position size calculation result."""
    size_usd: float
    size_units: float
    risk_per_trade_usd: float
    stop_loss_distance_percent: float


class PositionSizer:
    """
    Position sizing calculator using fixed percentage risk.

    Implements conservative position sizing:
    - Max 5% of capital per trade
    - Can use Kelly criterion for optimal sizing (optional)
    - Adjusts for volatility (optional)
    """

    def __init__(
        self,
        max_position_percent: float = 5.0,
        max_position_usd: float = 500.0,
        risk_per_trade_percent: float = 1.0
    ):
        """
        Initialize position sizer.

        Args:
            max_position_percent: Maximum position as % of capital.
            max_position_usd: Maximum position in absolute USD.
            risk_per_trade_percent: Max % of capital to risk per trade.
        """
        self.max_position_percent = max_position_percent
        self.max_position_usd = max_position_usd
        self.risk_per_trade_percent = risk_per_trade_percent

    def calculate_max_position(self, capital: float) -> float:
        """
        Calculate maximum position size.

        Args:
            capital: Total account capital.

        Returns:
            Maximum position size in USD.
        """
        percent_based = capital * (self.max_position_percent / 100)
        return min(percent_based, self.max_position_usd)

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float,
        signal_strength: float = 1.0
    ) -> PositionSize:
        """
        Calculate position size based on risk parameters.

        Uses the formula:
        Position Size = (Capital * Risk%) / (Stop Loss Distance%)

        Args:
            capital: Total account capital.
            entry_price: Planned entry price.
            stop_loss_price: Stop loss price.
            signal_strength: Signal strength (0.0-1.0) for size adjustment.

        Returns:
            PositionSize with calculation details.
        """
        # Calculate stop loss distance
        stop_loss_distance = abs(entry_price - stop_loss_price)
        stop_loss_distance_percent = (stop_loss_distance / entry_price) * 100

        if stop_loss_distance_percent <= 0:
            logger.warning("Invalid stop loss distance, using max position size")
            max_size = self.calculate_max_position(capital)
            return PositionSize(
                size_usd=max_size,
                size_units=max_size / entry_price,
                risk_per_trade_usd=capital * (self.risk_per_trade_percent / 100),
                stop_loss_distance_percent=0.0
            )

        # Calculate risk-based position size
        risk_amount = capital * (self.risk_per_trade_percent / 100)
        risk_based_size = risk_amount / (stop_loss_distance_percent / 100)

        # Apply max position limits
        max_size = self.calculate_max_position(capital)
        position_size = min(risk_based_size, max_size)

        # Adjust for signal strength (weaker signals = smaller positions)
        adjusted_size = position_size * max(0.5, signal_strength)

        logger.debug(
            f"Position size calculated: ${adjusted_size:.2f}",
            capital=capital,
            risk_based=risk_based_size,
            max_size=max_size,
            signal_strength=signal_strength,
            final_size=adjusted_size
        )

        return PositionSize(
            size_usd=adjusted_size,
            size_units=adjusted_size / entry_price,
            risk_per_trade_usd=risk_amount,
            stop_loss_distance_percent=stop_loss_distance_percent
        )

    def calculate_kelly_size(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly criterion.

        Kelly formula: f* = (bp - q) / b
        where:
        - b = win/loss ratio
        - p = probability of winning
        - q = probability of losing

        Args:
            capital: Total account capital.
            win_rate: Historical win rate (0.0-1.0).
            avg_win: Average winning trade amount.
            avg_loss: Average losing trade amount.
            fraction: Fraction of Kelly to use (default 1/4 Kelly).

        Returns:
            Position size in USD.
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return self.calculate_max_position(capital) * fraction

        # Calculate Kelly fraction
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        p = win_rate
        q = 1 - win_rate

        kelly = (b * p - q) / b

        # Apply fractional Kelly (safer)
        kelly *= fraction

        # Ensure non-negative and capped
        kelly = max(0, min(kelly, self.max_position_percent / 100))

        position_size = capital * kelly

        logger.debug(
            f"Kelly position size: ${position_size:.2f}",
            kelly_fraction=kelly,
            win_rate=win_rate,
            win_loss_ratio=b
        )

        return min(position_size, self.calculate_max_position(capital))

    def adjust_for_volatility(
        self,
        base_size: float,
        current_volatility: float,
        average_volatility: float
    ) -> float:
        """
        Adjust position size based on current volatility.

        Reduces position size when volatility is above average.

        Args:
            base_size: Base position size.
            current_volatility: Current volatility measure.
            average_volatility: Historical average volatility.

        Returns:
            Adjusted position size.
        """
        if average_volatility <= 0:
            return base_size

        vol_ratio = current_volatility / average_volatility

        # Reduce size when volatility is high
        if vol_ratio > 1.5:
            adjustment = 1 / vol_ratio
            adjusted_size = base_size * adjustment
            logger.debug(
                f"Volatility adjustment: ${base_size:.2f} -> ${adjusted_size:.2f}",
                vol_ratio=vol_ratio,
                adjustment=adjustment
            )
            return adjusted_size

        return base_size


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_percent: float,
    max_position_percent: float = 5.0
) -> float:
    """
    Convenience function to calculate position size.

    Args:
        capital: Total account capital.
        entry_price: Entry price.
        stop_loss_percent: Stop loss as percentage.
        max_position_percent: Max position as % of capital.

    Returns:
        Position size in USD.
    """
    sizer = PositionSizer(max_position_percent=max_position_percent)
    stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
    result = sizer.calculate_position_size(capital, entry_price, stop_loss_price)
    return result.size_usd
