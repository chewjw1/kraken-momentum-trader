"""
Base strategy interface for trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ..exchange.kraken_client import OHLC, Ticker


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"  # Open a short position
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class Signal:
    """Trading signal with metadata."""
    signal_type: SignalType
    pair: str
    strength: float  # 0.0 to 1.0, higher = stronger signal
    price: float
    timestamp: datetime
    reason: str
    indicators: dict = field(default_factory=dict)

    def is_actionable(self, min_strength: float = 0.5) -> bool:
        """Check if signal is strong enough to act on."""
        return (
            self.signal_type in (
                SignalType.BUY, SignalType.SELL, SignalType.SELL_SHORT,
                SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT
            )
            and self.strength >= min_strength
        )


@dataclass
class Position:
    """Current position information."""
    pair: str
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    size: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0

    def update_price(self, current_price: float) -> None:
        """Update position with current price."""
        self.current_price = current_price
        if self.side == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

        if self.entry_price > 0:
            self.unrealized_pnl_percent = (
                (current_price - self.entry_price) / self.entry_price * 100
            )
            if self.side == "short":
                self.unrealized_pnl_percent = -self.unrealized_pnl_percent


@dataclass
class MarketData:
    """Market data container for strategy analysis."""
    pair: str
    ticker: Optional[Ticker] = None
    ohlc: list[OHLC] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)

    @classmethod
    def from_ohlc(cls, pair: str, ohlc_data: list[OHLC], ticker: Optional[Ticker] = None) -> "MarketData":
        """Create MarketData from OHLC data."""
        return cls(
            pair=pair,
            ticker=ticker,
            ohlc=ohlc_data,
            prices=[candle.close for candle in ohlc_data],
            volumes=[candle.volume for candle in ohlc_data]
        )


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies analyze market data and generate trading signals
    based on their specific logic.
    """

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Strategy name for identification.
        """
        self.name = name
        self._is_initialized = False

    @abstractmethod
    def analyze(
        self,
        market_data: MarketData,
        current_position: Optional[Position] = None
    ) -> Signal:
        """
        Analyze market data and generate a trading signal.

        Args:
            market_data: Current market data.
            current_position: Current open position, if any.

        Returns:
            Trading signal.
        """
        pass

    @abstractmethod
    def check_trailing_stop(
        self,
        position: Position,
        current_price: float,
        peak_price: float,
        trailing_stop_active: bool
    ) -> tuple[bool, bool, str]:
        """
        Check trailing stop conditions.

        Args:
            position: Current position.
            current_price: Current market price.
            peak_price: Highest price reached since entry.
            trailing_stop_active: Whether trailing stop has been activated.

        Returns:
            Tuple of (should_close, should_activate_trailing, reason).
        """
        pass

    def should_close_position(
        self,
        position: Position,
        market_data: MarketData,
        peak_price: float = None,
        trailing_stop_active: bool = False
    ) -> tuple[bool, bool, str]:
        """
        Check if an open position should be closed.

        Args:
            position: Current position.
            market_data: Current market data.
            peak_price: Highest price reached (for trailing stop).
            trailing_stop_active: Whether trailing stop is active.

        Returns:
            Tuple of (should_close, should_activate_trailing, reason).
        """
        current_price = market_data.ticker.last if market_data.ticker else position.current_price

        if peak_price is None:
            peak_price = position.entry_price

        return self.check_trailing_stop(
            position, current_price, peak_price, trailing_stop_active
        )

    def reset(self) -> None:
        """Reset strategy state."""
        self._is_initialized = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
