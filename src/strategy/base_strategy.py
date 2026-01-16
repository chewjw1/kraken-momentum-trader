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
            self.signal_type in (SignalType.BUY, SignalType.SELL, SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT)
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
    def get_stop_loss_price(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price for a position.

        Args:
            entry_price: Position entry price.
            side: Position side ("long" or "short").

        Returns:
            Stop loss price.
        """
        pass

    @abstractmethod
    def get_take_profit_price(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price for a position.

        Args:
            entry_price: Position entry price.
            side: Position side ("long" or "short").

        Returns:
            Take profit price.
        """
        pass

    def should_close_position(
        self,
        position: Position,
        market_data: MarketData
    ) -> tuple[bool, str]:
        """
        Check if an open position should be closed.

        Args:
            position: Current position.
            market_data: Current market data.

        Returns:
            Tuple of (should_close, reason).
        """
        current_price = market_data.ticker.last if market_data.ticker else position.current_price

        # Check stop loss
        stop_loss = self.get_stop_loss_price(position.entry_price, position.side)
        if position.side == "long" and current_price <= stop_loss:
            return True, f"Stop loss hit: {current_price:.2f} <= {stop_loss:.2f}"
        if position.side == "short" and current_price >= stop_loss:
            return True, f"Stop loss hit: {current_price:.2f} >= {stop_loss:.2f}"

        # Check take profit
        take_profit = self.get_take_profit_price(position.entry_price, position.side)
        if position.side == "long" and current_price >= take_profit:
            return True, f"Take profit hit: {current_price:.2f} >= {take_profit:.2f}"
        if position.side == "short" and current_price <= take_profit:
            return True, f"Take profit hit: {current_price:.2f} <= {take_profit:.2f}"

        return False, ""

    def reset(self) -> None:
        """Reset strategy state."""
        self._is_initialized = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
