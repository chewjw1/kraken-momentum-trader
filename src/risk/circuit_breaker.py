"""
Circuit breaker for emergency trading halts.
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

from ..observability.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading paused (cooldown)
    EMERGENCY = "emergency"  # All positions to be closed


@dataclass
class CircuitBreakerState:
    """Current circuit breaker state information."""
    state: CircuitState
    consecutive_losses: int
    cooldown_until: Optional[datetime]
    trigger_reason: Optional[str]
    last_state_change: datetime


class CircuitBreaker:
    """
    Circuit breaker pattern for trading risk management.

    States:
    - CLOSED: Normal operation, trading allowed
    - OPEN: Cooldown period after consecutive losses
    - EMERGENCY: Critical condition, exit all positions

    Triggers:
    - Consecutive losses >= limit → OPEN state
    - Cooldown period expires → CLOSED state
    - Max drawdown exceeded → EMERGENCY state
    """

    def __init__(
        self,
        consecutive_loss_limit: int = 3,
        cooldown_hours: int = 4
    ):
        """
        Initialize circuit breaker.

        Args:
            consecutive_loss_limit: Number of consecutive losses before cooldown.
            cooldown_hours: Hours to pause trading after trigger.
        """
        self.consecutive_loss_limit = consecutive_loss_limit
        self.cooldown_hours = cooldown_hours

        # State
        self._state = CircuitState.CLOSED
        self._consecutive_losses = 0
        self._cooldown_until: Optional[datetime] = None
        self._trigger_reason: Optional[str] = None
        self._last_state_change = datetime.now(timezone.utc)

    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed.
        """
        self._check_cooldown_expired()
        return self._state == CircuitState.CLOSED

    def record_loss(self) -> None:
        """Record a losing trade."""
        self._consecutive_losses += 1

        logger.info(
            f"Loss recorded: {self._consecutive_losses}/{self.consecutive_loss_limit} consecutive",
            consecutive_losses=self._consecutive_losses,
            limit=self.consecutive_loss_limit
        )

        if self._consecutive_losses >= self.consecutive_loss_limit:
            self._trigger_cooldown()

    def record_win(self) -> None:
        """Record a winning trade."""
        if self._consecutive_losses > 0:
            logger.info(
                f"Win recorded, resetting consecutive loss counter from {self._consecutive_losses}",
                previous_losses=self._consecutive_losses
            )
        self._consecutive_losses = 0

    def trigger_emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop.

        Args:
            reason: Reason for emergency stop.
        """
        self._state = CircuitState.EMERGENCY
        self._trigger_reason = reason
        self._last_state_change = datetime.now(timezone.utc)

        logger.risk_event(
            event_type="emergency_stop",
            details=reason
        )

    def get_state(self) -> CircuitBreakerState:
        """
        Get current circuit breaker state.

        Returns:
            CircuitBreakerState with current status.
        """
        self._check_cooldown_expired()

        return CircuitBreakerState(
            state=self._state,
            consecutive_losses=self._consecutive_losses,
            cooldown_until=self._cooldown_until,
            trigger_reason=self._trigger_reason,
            last_state_change=self._last_state_change
        )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._consecutive_losses = 0
        self._cooldown_until = None
        self._trigger_reason = None
        self._last_state_change = datetime.now(timezone.utc)

        logger.info("Circuit breaker reset to CLOSED state")

    def _trigger_cooldown(self) -> None:
        """Trigger cooldown period."""
        self._state = CircuitState.OPEN
        self._cooldown_until = datetime.now(timezone.utc) + timedelta(hours=self.cooldown_hours)
        self._trigger_reason = f"{self._consecutive_losses} consecutive losses"
        self._last_state_change = datetime.now(timezone.utc)

        logger.risk_event(
            event_type="circuit_breaker_open",
            details=f"Trading paused until {self._cooldown_until.isoformat()}",
            consecutive_losses=self._consecutive_losses,
            cooldown_hours=self.cooldown_hours
        )

    def _check_cooldown_expired(self) -> None:
        """Check if cooldown period has expired."""
        if self._state == CircuitState.OPEN and self._cooldown_until:
            if datetime.now(timezone.utc) >= self._cooldown_until:
                logger.info(
                    "Cooldown expired, resetting circuit breaker",
                    cooldown_ended=self._cooldown_until.isoformat()
                )
                self._state = CircuitState.CLOSED
                self._consecutive_losses = 0
                self._cooldown_until = None
                self._trigger_reason = None
                self._last_state_change = datetime.now(timezone.utc)

    def time_until_ready(self) -> Optional[timedelta]:
        """
        Get time until circuit breaker is ready.

        Returns:
            Timedelta until ready, or None if already ready.
        """
        if self._state == CircuitState.CLOSED:
            return None

        if self._state == CircuitState.OPEN and self._cooldown_until:
            remaining = self._cooldown_until - datetime.now(timezone.utc)
            return remaining if remaining.total_seconds() > 0 else None

        # Emergency state doesn't have automatic recovery
        return None

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            "state": self._state.value,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "trigger_reason": self._trigger_reason,
            "last_state_change": self._last_state_change.isoformat(),
        }

    def from_dict(self, data: dict) -> None:
        """Restore state from dictionary."""
        self._state = CircuitState(data.get("state", "closed"))
        self._consecutive_losses = data.get("consecutive_losses", 0)
        self._trigger_reason = data.get("trigger_reason")

        if data.get("cooldown_until"):
            self._cooldown_until = datetime.fromisoformat(data["cooldown_until"])
        else:
            self._cooldown_until = None

        if data.get("last_state_change"):
            self._last_state_change = datetime.fromisoformat(data["last_state_change"])
        else:
            self._last_state_change = datetime.now(timezone.utc)

        # Check if cooldown has expired since last save
        self._check_cooldown_expired()
