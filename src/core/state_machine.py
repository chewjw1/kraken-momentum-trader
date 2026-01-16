"""
Trading state machine for managing trading lifecycle.
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
    IDLE = "idle"  # No position, waiting for signals
    ANALYZING = "analyzing"  # Analyzing market for entry
    ENTERING = "entering"  # Placing entry order
    IN_POSITION = "in_position"  # Holding a position
    EXITING = "exiting"  # Placing exit order
    ERROR = "error"  # Error state, requires intervention
    STOPPED = "stopped"  # Trading halted


@dataclass
class TradingPosition:
    """Current trading position."""
    pair: str
    side: str  # "long" or "short"
    entry_price: float
    size: float
    entry_time: datetime
    order_id: str
    stop_loss: float
    take_profit: float


@dataclass
class StateMachineState:
    """Complete state machine state."""
    current_state: TradingState
    previous_state: Optional[TradingState]
    position: Optional[TradingPosition]
    last_state_change: datetime
    error_message: Optional[str]
    pending_order_id: Optional[str]


class TradingStateMachine:
    """
    State machine for managing trading lifecycle.

    States:
    - INITIALIZING: Starting up, loading state
    - IDLE: No position, waiting for entry signals
    - ANALYZING: Evaluating market conditions
    - ENTERING: Placing entry order
    - IN_POSITION: Holding an open position
    - EXITING: Closing position
    - ERROR: Error occurred, needs intervention
    - STOPPED: Trading manually stopped

    Transitions:
    - INITIALIZING -> IDLE: Initialization complete
    - IDLE -> ANALYZING: Market check triggered
    - ANALYZING -> ENTERING: Entry signal detected
    - ANALYZING -> IDLE: No signal
    - ENTERING -> IN_POSITION: Order filled
    - ENTERING -> ERROR: Order failed
    - IN_POSITION -> EXITING: Exit signal or stop/take profit
    - EXITING -> IDLE: Exit complete
    - EXITING -> ERROR: Exit failed
    - Any -> ERROR: Unhandled error
    - Any -> STOPPED: Manual stop
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        TradingState.INITIALIZING: [TradingState.IDLE, TradingState.ERROR, TradingState.STOPPED],
        TradingState.IDLE: [TradingState.ANALYZING, TradingState.ERROR, TradingState.STOPPED],
        TradingState.ANALYZING: [TradingState.ENTERING, TradingState.IDLE, TradingState.ERROR, TradingState.STOPPED],
        TradingState.ENTERING: [TradingState.IN_POSITION, TradingState.IDLE, TradingState.ERROR, TradingState.STOPPED],
        TradingState.IN_POSITION: [TradingState.EXITING, TradingState.ERROR, TradingState.STOPPED],
        TradingState.EXITING: [TradingState.IDLE, TradingState.IN_POSITION, TradingState.ERROR, TradingState.STOPPED],
        TradingState.ERROR: [TradingState.IDLE, TradingState.STOPPED],
        TradingState.STOPPED: [TradingState.IDLE],
    }

    def __init__(self):
        """Initialize the state machine."""
        self._state = TradingState.INITIALIZING
        self._previous_state: Optional[TradingState] = None
        self._position: Optional[TradingPosition] = None
        self._last_state_change = datetime.now(timezone.utc)
        self._error_message: Optional[str] = None
        self._pending_order_id: Optional[str] = None
        self._state_listeners: list = []

    @property
    def state(self) -> TradingState:
        """Get current state."""
        return self._state

    @property
    def position(self) -> Optional[TradingPosition]:
        """Get current position."""
        return self._position

    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self._position is not None

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
            position=self._position.pair if self._position else None
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
        return self.transition_to(TradingState.IDLE, "No entry signal")

    def open_position(
        self,
        pair: str,
        side: str,
        entry_price: float,
        size: float,
        order_id: str,
        stop_loss: float,
        take_profit: float
    ) -> bool:
        """
        Record an opened position.

        Args:
            pair: Trading pair.
            side: Position side ("long" or "short").
            entry_price: Entry price.
            size: Position size.
            order_id: Order ID.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
        """
        self._position = TradingPosition(
            pair=pair,
            side=side,
            entry_price=entry_price,
            size=size,
            entry_time=datetime.now(timezone.utc),
            order_id=order_id,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        return self.transition_to(
            TradingState.IN_POSITION,
            f"Position opened: {side} {size} {pair} @ {entry_price}"
        )

    def start_exit(self, reason: str = "") -> bool:
        """Begin exiting current position."""
        if not self._position:
            logger.warning("Cannot exit: no position")
            return False
        return self.transition_to(TradingState.EXITING, reason or "Exiting position")

    def close_position(self, exit_price: float, pnl: float) -> TradingPosition:
        """
        Record a closed position.

        Args:
            exit_price: Exit price.
            pnl: Realized P&L.

        Returns:
            The closed position.
        """
        closed_position = self._position
        self._position = None

        self.transition_to(
            TradingState.IDLE,
            f"Position closed @ {exit_price}, P&L: ${pnl:.2f}"
        )

        return closed_position

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
        return self.transition_to(TradingState.IDLE, "Error recovery")

    def stop(self, reason: str = "Manual stop") -> bool:
        """Stop trading."""
        return self.transition_to(TradingState.STOPPED, reason)

    def resume(self) -> bool:
        """Resume trading from stopped state."""
        if self._state != TradingState.STOPPED:
            return False
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
            position=self._position,
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
        }

        if self._position:
            data["position"] = {
                "pair": self._position.pair,
                "side": self._position.side,
                "entry_price": self._position.entry_price,
                "size": self._position.size,
                "entry_time": self._position.entry_time.isoformat(),
                "order_id": self._position.order_id,
                "stop_loss": self._position.stop_loss,
                "take_profit": self._position.take_profit,
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

        if data.get("position"):
            pos = data["position"]
            self._position = TradingPosition(
                pair=pos["pair"],
                side=pos["side"],
                entry_price=pos["entry_price"],
                size=pos["size"],
                entry_time=datetime.fromisoformat(pos["entry_time"]),
                order_id=pos["order_id"],
                stop_loss=pos["stop_loss"],
                take_profit=pos["take_profit"],
            )
        else:
            self._position = None

        logger.info(
            f"State restored: {self._state.value}",
            has_position=self._position is not None
        )
