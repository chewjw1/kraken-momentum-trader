"""
Trading state machine for managing trading lifecycle.
Supports multiple simultaneous positions across different pairs.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ..observability.logger import get_logger

logger = get_logger(__name__)


class TradingState(Enum):
    """Trading lifecycle states."""
    INITIALIZING = "initializing"
    IDLE = "idle"  # Waiting for signals (may have positions)
    ANALYZING = "analyzing"  # Analyzing market for entry
    ENTERING = "entering"  # Placing entry order
    IN_POSITION = "in_position"  # Holding one or more positions
    EXITING = "exiting"  # Placing exit order
    ERROR = "error"  # Error state, requires intervention
    STOPPED = "stopped"  # Trading halted


@dataclass
class PositionEntry:
    """Individual entry in a position (for Martingale support)."""
    price: float
    size: float
    size_usd: float
    entry_time: datetime
    order_id: str


@dataclass
class TradingPosition:
    """Current trading position with multi-entry support for Martingale."""
    pair: str
    side: str  # "long" or "short"
    entries: list  # List of PositionEntry objects
    peak_price: float  # Highest price reached (for trailing stop)
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
    def entry_time(self) -> datetime:
        """Time of first entry."""
        if not self.entries:
            return datetime.now(timezone.utc)
        return self.entries[0].entry_time

    @property
    def order_id(self) -> str:
        """Order ID of first entry (for compatibility)."""
        if not self.entries:
            return ""
        return self.entries[0].order_id

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


@dataclass
class StateMachineState:
    """Complete state machine state."""
    current_state: TradingState
    previous_state: Optional[TradingState]
    positions: dict  # pair -> TradingPosition
    last_state_change: datetime
    error_message: Optional[str]
    pending_order_id: Optional[str]


class TradingStateMachine:
    """
    State machine for managing trading lifecycle.
    Supports multiple simultaneous positions across different pairs.

    States:
    - INITIALIZING: Starting up, loading state
    - IDLE: Waiting for entry signals (may have open positions)
    - ANALYZING: Evaluating market conditions
    - ENTERING: Placing entry order
    - IN_POSITION: Holding one or more open positions
    - EXITING: Closing a position
    - ERROR: Error occurred, needs intervention
    - STOPPED: Trading manually stopped

    Transitions:
    - INITIALIZING -> IDLE: Initialization complete
    - IDLE -> ANALYZING: Market check triggered
    - ANALYZING -> ENTERING: Entry signal detected
    - ANALYZING -> IDLE: No signal
    - ENTERING -> IN_POSITION: Order filled
    - ENTERING -> IDLE: Order filled but no positions remain
    - ENTERING -> ERROR: Order failed
    - IN_POSITION -> EXITING: Exit signal or stop/take profit
    - IN_POSITION -> ANALYZING: Check other pairs
    - EXITING -> IDLE: All positions closed
    - EXITING -> IN_POSITION: Some positions remain
    - EXITING -> ERROR: Exit failed
    - Any -> ERROR: Unhandled error
    - Any -> STOPPED: Manual stop
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        TradingState.INITIALIZING: [TradingState.IDLE, TradingState.IN_POSITION, TradingState.ERROR, TradingState.STOPPED],
        TradingState.IDLE: [TradingState.ANALYZING, TradingState.IN_POSITION, TradingState.ERROR, TradingState.STOPPED],
        TradingState.ANALYZING: [TradingState.ENTERING, TradingState.IDLE, TradingState.IN_POSITION, TradingState.ERROR, TradingState.STOPPED],
        TradingState.ENTERING: [TradingState.IN_POSITION, TradingState.IDLE, TradingState.ERROR, TradingState.STOPPED],
        TradingState.IN_POSITION: [TradingState.EXITING, TradingState.ANALYZING, TradingState.IDLE, TradingState.ERROR, TradingState.STOPPED],
        TradingState.EXITING: [TradingState.IDLE, TradingState.IN_POSITION, TradingState.ERROR, TradingState.STOPPED],
        TradingState.ERROR: [TradingState.IDLE, TradingState.IN_POSITION, TradingState.STOPPED],
        TradingState.STOPPED: [TradingState.IDLE, TradingState.IN_POSITION],
    }

    def __init__(self):
        """Initialize the state machine."""
        self._state = TradingState.INITIALIZING
        self._previous_state: Optional[TradingState] = None
        self._positions: dict[str, TradingPosition] = {}  # pair -> position
        self._last_state_change = datetime.now(timezone.utc)
        self._error_message: Optional[str] = None
        self._pending_order_id: Optional[str] = None
        self._state_listeners: list = []

    @property
    def state(self) -> TradingState:
        """Get current state."""
        return self._state

    @property
    def positions(self) -> dict[str, TradingPosition]:
        """Get all open positions."""
        return self._positions

    @property
    def position(self) -> Optional[TradingPosition]:
        """Get first position (for backward compatibility)."""
        if self._positions:
            return next(iter(self._positions.values()))
        return None

    @property
    def has_position(self) -> bool:
        """Check if there are any open positions."""
        return len(self._positions) > 0

    def has_position_for_pair(self, pair: str) -> bool:
        """Check if there's an open position for a specific pair."""
        return pair in self._positions

    def get_position(self, pair: str) -> Optional[TradingPosition]:
        """Get position for a specific pair."""
        return self._positions.get(pair)

    @property
    def total_exposure_usd(self) -> float:
        """Get total USD exposure across all positions."""
        return sum(pos.total_cost_usd for pos in self._positions.values())

    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len(self._positions)

    def transition_to(self, new_state: TradingState, reason: str = "") -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state.
            reason: Reason for transition.

        Returns:
            True if transition was successful.
        """
        if new_state not in self.VALID_TRANSITIONS.get(self._state, []):
            logger.warning(
                f"Invalid state transition: {self._state.value} -> {new_state.value}",
                current_state=self._state.value,
                target_state=new_state.value
            )
            return False

        self._previous_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now(timezone.utc)

        if new_state != TradingState.ERROR:
            self._error_message = None

        logger.info(
            f"State transition: {self._previous_state.value} -> {new_state.value}",
            reason=reason,
            open_positions=list(self._positions.keys())
        )

        # Notify listeners
        for listener in self._state_listeners:
            try:
                listener(self._previous_state, new_state, reason)
            except Exception as e:
                logger.error(f"State listener error: {e}")

        return True

    def initialize_complete(self) -> bool:
        """Mark initialization as complete."""
        # Go to IN_POSITION if we restored positions, otherwise IDLE
        if self._positions:
            return self.transition_to(TradingState.IN_POSITION, "Initialization complete with open positions")
        return self.transition_to(TradingState.IDLE, "Initialization complete")

    def start_analysis(self) -> bool:
        """Begin market analysis."""
        return self.transition_to(TradingState.ANALYZING, "Starting market analysis")

    def analysis_complete(self, has_signal: bool) -> bool:
        """
        Complete analysis phase.

        Args:
            has_signal: Whether an entry signal was found.
        """
        if has_signal:
            return self.transition_to(TradingState.ENTERING, "Entry signal detected")
        # Return to IN_POSITION if we have positions, otherwise IDLE
        if self._positions:
            return self.transition_to(TradingState.IN_POSITION, "No entry signal, maintaining positions")
        return self.transition_to(TradingState.IDLE, "No entry signal")

    def open_position(
        self,
        pair: str,
        side: str,
        entry_price: float,
        size: float,
        size_usd: float,
        order_id: str
    ) -> bool:
        """
        Record an opened position.

        Args:
            pair: Trading pair.
            side: Position side ("long" or "short").
            entry_price: Entry price.
            size: Position size in units.
            size_usd: Position size in USD.
            order_id: Order ID.
        """
        if pair in self._positions:
            logger.warning(f"Position already exists for {pair}, use add_to_position instead")
            return False

        initial_entry = PositionEntry(
            price=entry_price,
            size=size,
            size_usd=size_usd,
            entry_time=datetime.now(timezone.utc),
            order_id=order_id
        )

        self._positions[pair] = TradingPosition(
            pair=pair,
            side=side,
            entries=[initial_entry],
            peak_price=entry_price,  # Initialize peak at entry
            trailing_stop_active=False
        )

        # Only transition if not already in position
        if self._state != TradingState.IN_POSITION:
            return self.transition_to(
                TradingState.IN_POSITION,
                f"Position opened: {side} {size:.6f} {pair} @ {entry_price:.2f} (total positions: {len(self._positions)})"
            )
        else:
            logger.info(
                f"Position opened: {side} {size:.6f} {pair} @ {entry_price:.2f} (total positions: {len(self._positions)})"
            )
            return True

    def add_to_position(
        self,
        pair: str,
        entry_price: float,
        size: float,
        size_usd: float,
        order_id: str
    ) -> bool:
        """
        Add to existing position (Martingale).

        Args:
            pair: Trading pair.
            entry_price: Entry price for this add-on.
            size: Position size in units.
            size_usd: Position size in USD.
            order_id: Order ID.

        Returns:
            True if add was successful.
        """
        if pair not in self._positions:
            logger.warning(f"Cannot add to position: no position exists for {pair}")
            return False

        position = self._positions[pair]

        new_entry = PositionEntry(
            price=entry_price,
            size=size,
            size_usd=size_usd,
            entry_time=datetime.now(timezone.utc),
            order_id=order_id
        )

        position.entries.append(new_entry)

        # Reset trailing stop when adding (new average price)
        position.trailing_stop_active = False
        # Update peak to current price if adding on dip
        if entry_price > position.peak_price:
            position.peak_price = entry_price

        logger.info(
            f"Added to position: +{size:.6f} @ {entry_price:.2f} "
            f"(total entries: {position.num_entries}, "
            f"avg price: {position.entry_price:.2f})",
            pair=pair,
            num_entries=position.num_entries,
            total_size=position.size,
            avg_entry=position.entry_price
        )

        return True

    def update_peak_price(self, pair: str, current_price: float) -> bool:
        """
        Update peak price for trailing stop tracking.

        Args:
            pair: Trading pair.
            current_price: Current market price.

        Returns:
            True if peak was updated.
        """
        if pair not in self._positions:
            return False

        position = self._positions[pair]

        if position.side == "long" and current_price > position.peak_price:
            position.peak_price = current_price
            return True
        elif position.side == "short" and current_price < position.peak_price:
            position.peak_price = current_price
            return True

        return False

    def activate_trailing_stop(self, pair: str) -> None:
        """Activate the trailing stop for a specific position."""
        if pair in self._positions:
            self._positions[pair].trailing_stop_active = True
            logger.info(
                f"Trailing stop activated at peak price {self._positions[pair].peak_price}",
                pair=pair
            )

    def start_exit(self, pair: str, reason: str = "") -> bool:
        """Begin exiting a specific position."""
        if pair not in self._positions:
            logger.warning(f"Cannot exit: no position for {pair}")
            return False
        return self.transition_to(TradingState.EXITING, reason or f"Exiting {pair} position")

    def close_position(self, pair: str, exit_price: float, pnl: float) -> Optional[TradingPosition]:
        """
        Record a closed position.

        Args:
            pair: Trading pair.
            exit_price: Exit price.
            pnl: Realized P&L.

        Returns:
            The closed position or None if not found.
        """
        if pair not in self._positions:
            logger.warning(f"Cannot close: no position for {pair}")
            return None

        closed_position = self._positions.pop(pair)

        # Transition based on remaining positions
        if self._positions:
            # Already in IN_POSITION state, just log
            logger.info(
                f"Closed {pair} @ {exit_price}, P&L: ${pnl:.2f} (remaining positions: {len(self._positions)})"
            )
        else:
            self.transition_to(
                TradingState.IDLE,
                f"Closed {pair} @ {exit_price}, P&L: ${pnl:.2f} (no positions remaining)"
            )

        return closed_position

    def close_all_positions(self) -> list[TradingPosition]:
        """
        Close all positions (for emergency exit).

        Returns:
            List of closed positions.
        """
        closed = list(self._positions.values())
        self._positions.clear()
        self.transition_to(TradingState.IDLE, "All positions closed")
        return closed

    def set_error(self, error_message: str) -> bool:
        """
        Transition to error state.

        Args:
            error_message: Error description.
        """
        self._error_message = error_message
        return self.transition_to(TradingState.ERROR, error_message)

    def recover_from_error(self) -> bool:
        """Attempt to recover from error state."""
        if self._state != TradingState.ERROR:
            return False
        if self._positions:
            return self.transition_to(TradingState.IN_POSITION, "Error recovery with positions")
        return self.transition_to(TradingState.IDLE, "Error recovery")

    def stop(self, reason: str = "Manual stop") -> bool:
        """Stop trading."""
        return self.transition_to(TradingState.STOPPED, reason)

    def resume(self) -> bool:
        """Resume trading from stopped state."""
        if self._state != TradingState.STOPPED:
            return False
        if self._positions:
            return self.transition_to(TradingState.IN_POSITION, "Resuming trading with positions")
        return self.transition_to(TradingState.IDLE, "Resuming trading")

    def set_pending_order(self, order_id: str) -> None:
        """Set pending order ID."""
        self._pending_order_id = order_id

    def clear_pending_order(self) -> None:
        """Clear pending order ID."""
        self._pending_order_id = None

    def add_state_listener(self, listener) -> None:
        """
        Add a state change listener.

        Args:
            listener: Callback function(previous_state, new_state, reason).
        """
        self._state_listeners.append(listener)

    def get_full_state(self) -> StateMachineState:
        """Get complete state machine state."""
        return StateMachineState(
            current_state=self._state,
            previous_state=self._previous_state,
            positions=self._positions.copy(),
            last_state_change=self._last_state_change,
            error_message=self._error_message,
            pending_order_id=self._pending_order_id
        )

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        data = {
            "state": self._state.value,
            "previous_state": self._previous_state.value if self._previous_state else None,
            "last_state_change": self._last_state_change.isoformat(),
            "error_message": self._error_message,
            "pending_order_id": self._pending_order_id,
            "positions": {}
        }

        for pair, position in self._positions.items():
            data["positions"][pair] = {
                "pair": position.pair,
                "side": position.side,
                "entries": [
                    {
                        "price": e.price,
                        "size": e.size,
                        "size_usd": e.size_usd,
                        "entry_time": e.entry_time.isoformat(),
                        "order_id": e.order_id,
                    }
                    for e in position.entries
                ],
                "peak_price": position.peak_price,
                "trailing_stop_active": position.trailing_stop_active,
            }

        return data

    def from_dict(self, data: dict) -> None:
        """Restore state from dictionary."""
        self._state = TradingState(data.get("state", "initializing"))
        self._previous_state = (
            TradingState(data["previous_state"])
            if data.get("previous_state")
            else None
        )
        self._last_state_change = (
            datetime.fromisoformat(data["last_state_change"])
            if data.get("last_state_change")
            else datetime.now(timezone.utc)
        )
        self._error_message = data.get("error_message")
        self._pending_order_id = data.get("pending_order_id")

        # Load positions
        self._positions.clear()

        # Handle new multi-position format
        if "positions" in data and isinstance(data["positions"], dict):
            for pair, pos in data["positions"].items():
                entries = [
                    PositionEntry(
                        price=e["price"],
                        size=e["size"],
                        size_usd=e.get("size_usd", e["price"] * e["size"]),
                        entry_time=datetime.fromisoformat(e["entry_time"]),
                        order_id=e["order_id"],
                    )
                    for e in pos["entries"]
                ]

                self._positions[pair] = TradingPosition(
                    pair=pos["pair"],
                    side=pos["side"],
                    entries=entries,
                    peak_price=pos.get("peak_price", entries[0].price if entries else 0),
                    trailing_stop_active=pos.get("trailing_stop_active", False),
                )

        # Handle legacy single-position format
        elif "position" in data and data["position"]:
            pos = data["position"]

            if "entries" in pos:
                entries = [
                    PositionEntry(
                        price=e["price"],
                        size=e["size"],
                        size_usd=e.get("size_usd", e["price"] * e["size"]),
                        entry_time=datetime.fromisoformat(e["entry_time"]),
                        order_id=e["order_id"],
                    )
                    for e in pos["entries"]
                ]
            else:
                # Legacy format - convert single entry to entries list
                entries = [
                    PositionEntry(
                        price=pos["entry_price"],
                        size=pos["size"],
                        size_usd=pos["entry_price"] * pos["size"],
                        entry_time=datetime.fromisoformat(pos["entry_time"]),
                        order_id=pos["order_id"],
                    )
                ]

            pair = pos["pair"]
            self._positions[pair] = TradingPosition(
                pair=pair,
                side=pos["side"],
                entries=entries,
                peak_price=pos.get("peak_price", entries[0].price if entries else 0),
                trailing_stop_active=pos.get("trailing_stop_active", False),
            )

        logger.info(
            f"State restored: {self._state.value}",
            open_positions=list(self._positions.keys())
        )
