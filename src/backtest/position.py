"""
Simulated position for backtesting.
Mirrors TradingPosition with multi-entry Martingale support.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class SimulatedEntry:
    """Individual entry in a simulated position."""
    price: float
    size: float
    size_usd: float
    entry_time: datetime
    fee: float = 0.0


@dataclass
class SimulatedPosition:
    """
    Simulated trading position with multi-entry support for Martingale.

    This mirrors the TradingPosition from the live trader but is designed
    for backtest simulation without order IDs.
    """
    pair: str
    side: str  # "long" or "short"
    entries: List[SimulatedEntry] = field(default_factory=list)
    peak_price: float = 0.0
    trailing_stop_active: bool = False

    @property
    def entry_price(self) -> float:
        """Weighted average entry price across all entries."""
        if not self.entries:
            return 0.0
        total_size = sum(e.size for e in self.entries)
        if total_size == 0:
            return 0.0
        weighted_sum = sum(e.price * e.size for e in self.entries)
        return weighted_sum / total_size

    @property
    def size(self) -> float:
        """Total position size across all entries."""
        return sum(e.size for e in self.entries)

    @property
    def total_cost_usd(self) -> float:
        """Total USD invested across all entries."""
        return sum(e.size_usd for e in self.entries)

    @property
    def total_fees(self) -> float:
        """Total fees paid across all entries."""
        return sum(e.fee for e in self.entries)

    @property
    def entry_time(self) -> datetime:
        """Time of first entry."""
        if not self.entries:
            return datetime.now(timezone.utc)
        return self.entries[0].entry_time

    @property
    def num_entries(self) -> int:
        """Number of entries in position."""
        return len(self.entries)

    @property
    def last_entry_size_usd(self) -> float:
        """Size of the most recent entry in USD."""
        if not self.entries:
            return 0.0
        return self.entries[-1].size_usd

    def add_entry(
        self,
        price: float,
        size: float,
        size_usd: float,
        entry_time: datetime,
        fee: float = 0.0
    ) -> None:
        """
        Add a new entry to the position.

        Args:
            price: Entry price.
            size: Position size in base currency.
            size_usd: Position size in USD.
            entry_time: Time of entry.
            fee: Transaction fee.
        """
        entry = SimulatedEntry(
            price=price,
            size=size,
            size_usd=size_usd,
            entry_time=entry_time,
            fee=fee
        )
        self.entries.append(entry)

        # Reset trailing stop when adding (new average price)
        self.trailing_stop_active = False

        # Update peak if entry is higher
        if price > self.peak_price:
            self.peak_price = price

    def update_peak_price(self, current_price: float) -> bool:
        """
        Update peak price for trailing stop tracking.

        Args:
            current_price: Current market price.

        Returns:
            True if peak was updated.
        """
        if self.side == "long" and current_price > self.peak_price:
            self.peak_price = current_price
            return True
        elif self.side == "short" and current_price < self.peak_price:
            self.peak_price = current_price
            return True
        return False

    def activate_trailing_stop(self) -> None:
        """Activate the trailing stop."""
        self.trailing_stop_active = True

    def calculate_pnl(self, exit_price: float, exit_fee: float = 0.0) -> tuple[float, float]:
        """
        Calculate P&L for closing at given price.

        Args:
            exit_price: Price to exit at.
            exit_fee: Fee for exit transaction.

        Returns:
            Tuple of (gross_pnl, net_pnl after fees).
        """
        if self.side == "long":
            gross_pnl = (exit_price - self.entry_price) * self.size
        else:
            gross_pnl = (self.entry_price - exit_price) * self.size

        total_fees = self.total_fees + exit_fee
        net_pnl = gross_pnl - total_fees

        return gross_pnl, net_pnl

    def calculate_pnl_percent(self, exit_price: float) -> float:
        """
        Calculate P&L percentage.

        Args:
            exit_price: Price to exit at.

        Returns:
            P&L as percentage.
        """
        if self.entry_price == 0:
            return 0.0

        if self.side == "long":
            return ((exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - exit_price) / self.entry_price) * 100

    def to_dict(self) -> dict:
        """Serialize position to dictionary."""
        return {
            "pair": self.pair,
            "side": self.side,
            "entries": [
                {
                    "price": e.price,
                    "size": e.size,
                    "size_usd": e.size_usd,
                    "entry_time": e.entry_time.isoformat(),
                    "fee": e.fee,
                }
                for e in self.entries
            ],
            "peak_price": self.peak_price,
            "trailing_stop_active": self.trailing_stop_active,
            "entry_price": self.entry_price,
            "total_size": self.size,
            "total_cost_usd": self.total_cost_usd,
            "num_entries": self.num_entries,
        }


@dataclass
class CompletedTrade:
    """A completed backtest trade with entry and exit details."""
    trade_id: str
    pair: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    net_pnl: float
    pnl_percent: float
    entry_fees: float
    exit_fee: float
    num_entries: int  # Number of Martingale entries
    exit_reason: str

    @property
    def is_winner(self) -> bool:
        """Check if this trade was profitable."""
        return self.net_pnl > 0

    def to_dict(self) -> dict:
        """Serialize trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "pnl_percent": self.pnl_percent,
            "entry_fees": self.entry_fees,
            "exit_fee": self.exit_fee,
            "num_entries": self.num_entries,
            "exit_reason": self.exit_reason,
            "is_winner": self.is_winner,
        }
