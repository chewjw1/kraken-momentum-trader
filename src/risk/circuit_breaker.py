"""
Drawdown-based circuit breaker for emergency trading halts.

Two layers:
- Per-pair: tracks drawdown from each pair's peak equity contribution.
  Pauses a single pair when its drawdown exceeds a threshold.
- Global: tracks total account drawdown from peak equity.
  Halts all trading when the account drawdown is too large.

This replaces the old consecutive-loss counter which was too fragile
for multi-pair trading (3 independent losses across 8 pairs is normal
variance, not a systemic problem).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, Dict

from ..observability.logger import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Trading paused (cooldown)
    EMERGENCY = "emergency"  # All positions to be closed


@dataclass
class CircuitBreakerState:
    """Current circuit breaker state information."""
    state: CircuitState
    consecutive_losses: int
    cooldown_until: Optional[datetime]
    trigger_reason: Optional[str]
    last_state_change: datetime


@dataclass
class PairDrawdownState:
    """Drawdown tracking for a single pair."""
    peak_pnl: float = 0.0
    current_pnl: float = 0.0
    is_paused: bool = False
    paused_until: Optional[datetime] = None
    pause_reason: Optional[str] = None

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak as a positive percentage (of initial capital)."""
        if self.peak_pnl <= 0:
            # Never been profitable -- drawdown is the total loss
            return abs(min(self.current_pnl, 0.0))
        dd = self.peak_pnl - self.current_pnl
        return max(dd, 0.0)


class CircuitBreaker:
    """
    Drawdown-based circuit breaker with per-pair and global tracking.

    States:
    - CLOSED: Normal operation, trading allowed
    - OPEN: Cooldown period after drawdown threshold breached
    - EMERGENCY: Critical condition, exit all positions

    Triggers:
    - Per-pair drawdown exceeds pair_max_drawdown_pct → pair paused
    - Global account drawdown exceeds global_max_drawdown_pct → all paused
    - Consecutive losses still tracked as secondary signal
    """

    def __init__(
        self,
        # Global drawdown settings
        global_max_drawdown_pct: float = 5.0,   # Halt all if account down 5% from peak
        cooldown_hours: int = 4,
        # Per-pair drawdown settings
        pair_max_drawdown_pct: float = 3.0,      # Pause pair if its P&L drops 3% from peak
        pair_cooldown_hours: float = 2.0,
        # Legacy consecutive loss support (secondary signal)
        consecutive_loss_limit: int = 5,         # Raised from 3 -- less trigger-happy
        # Capital reference
        initial_capital: float = 10000.0,
    ):
        self.global_max_drawdown_pct = global_max_drawdown_pct
        self.cooldown_hours = cooldown_hours
        self.pair_max_drawdown_pct = pair_max_drawdown_pct
        self.pair_cooldown_hours = pair_cooldown_hours
        self.consecutive_loss_limit = consecutive_loss_limit
        self.initial_capital = initial_capital

        # Global state
        self._state = CircuitState.CLOSED
        self._cooldown_until: Optional[datetime] = None
        self._trigger_reason: Optional[str] = None
        self._last_state_change = datetime.now(timezone.utc)

        # Global drawdown tracking
        self._peak_equity = initial_capital
        self._current_equity = initial_capital

        # Legacy consecutive loss counter (secondary)
        self._consecutive_losses = 0

        # Per-pair drawdown tracking
        self._pair_states: Dict[str, PairDrawdownState] = {}

    # ── Global interface ──────────────────────────────────────────

    def is_trading_allowed(self) -> bool:
        """Check if trading is globally allowed."""
        self._check_cooldown_expired()
        return self._state == CircuitState.CLOSED

    def is_pair_allowed(self, pair: str) -> bool:
        """
        Check if a specific pair is allowed to trade.

        A pair is blocked if it's individually paused OR if the global
        circuit breaker is open.
        """
        if not self.is_trading_allowed():
            return False

        ps = self._pair_states.get(pair)
        if ps and ps.is_paused:
            now = datetime.now(timezone.utc)
            if ps.paused_until and now >= ps.paused_until:
                # Cooldown expired -- un-pause and reset peak so the
                # pair doesn't immediately re-trigger on the next loss.
                ps.is_paused = False
                ps.paused_until = None
                ps.pause_reason = None
                ps.peak_pnl = ps.current_pnl
                logger.info(f"Pair {pair} un-paused (cooldown expired)")
                return True
            return False

        return True

    def record_trade(self, pair: str, pnl_usd: float) -> None:
        """
        Record a completed trade with its P&L.

        Updates both per-pair and global drawdown tracking.

        Args:
            pair: Trading pair.
            pnl_usd: Trade profit/loss in USD.
        """
        # Update global equity
        self._current_equity += pnl_usd
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        # Update per-pair tracking
        if pair not in self._pair_states:
            self._pair_states[pair] = PairDrawdownState()
        ps = self._pair_states[pair]
        ps.current_pnl += pnl_usd
        if ps.current_pnl > ps.peak_pnl:
            ps.peak_pnl = ps.current_pnl

        # Legacy consecutive loss counter
        if pnl_usd > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Check global drawdown
        global_dd_pct = self._global_drawdown_pct()
        if global_dd_pct >= self.global_max_drawdown_pct:
            self._trigger_cooldown(
                f"Global drawdown {global_dd_pct:.1f}% >= {self.global_max_drawdown_pct}%"
            )
            return

        # Check consecutive losses (secondary)
        if self._consecutive_losses >= self.consecutive_loss_limit:
            self._trigger_cooldown(
                f"{self._consecutive_losses} consecutive losses"
            )
            return

        # Check per-pair drawdown
        pair_dd = ps.drawdown_pct
        pair_dd_pct = (pair_dd / self.initial_capital) * 100
        if pair_dd_pct >= self.pair_max_drawdown_pct and not ps.is_paused:
            self._pause_pair(
                pair,
                f"Pair drawdown ${pair_dd:.2f} ({pair_dd_pct:.1f}% of capital) "
                f">= {self.pair_max_drawdown_pct}%"
            )

        logger.info(
            f"Trade recorded in circuit breaker",
            pair=pair,
            pnl_usd=f"${pnl_usd:.2f}",
            global_dd=f"{global_dd_pct:.2f}%",
            pair_dd=f"{pair_dd_pct:.2f}%",
            consecutive_losses=self._consecutive_losses,
        )

    # Legacy interface -- kept for backward compatibility
    def record_loss(self) -> None:
        """Legacy: record a losing trade (prefer record_trade)."""
        self._consecutive_losses += 1
        if self._consecutive_losses >= self.consecutive_loss_limit:
            self._trigger_cooldown(
                f"{self._consecutive_losses} consecutive losses"
            )

    def record_win(self) -> None:
        """Legacy: record a winning trade (prefer record_trade)."""
        self._consecutive_losses = 0

    def update_equity(self, current_equity: float) -> None:
        """
        Update current equity from external source (e.g. portfolio value).

        Useful for tracking unrealized P&L drawdown too.
        """
        self._current_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        global_dd_pct = self._global_drawdown_pct()
        if global_dd_pct >= self.global_max_drawdown_pct:
            self._trigger_cooldown(
                f"Global equity drawdown {global_dd_pct:.1f}% "
                f">= {self.global_max_drawdown_pct}%"
            )

    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop -- exit all positions."""
        self._state = CircuitState.EMERGENCY
        self._trigger_reason = reason
        self._last_state_change = datetime.now(timezone.utc)

        logger.risk_event(
            event_type="emergency_stop",
            details=reason
        )

    def get_state(self) -> CircuitBreakerState:
        """Get current global circuit breaker state."""
        self._check_cooldown_expired()
        return CircuitBreakerState(
            state=self._state,
            consecutive_losses=self._consecutive_losses,
            cooldown_until=self._cooldown_until,
            trigger_reason=self._trigger_reason,
            last_state_change=self._last_state_change,
        )

    def get_pair_state(self, pair: str) -> Optional[PairDrawdownState]:
        """Get drawdown state for a specific pair."""
        return self._pair_states.get(pair)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._consecutive_losses = 0
        self._cooldown_until = None
        self._trigger_reason = None
        self._last_state_change = datetime.now(timezone.utc)

        logger.info("Circuit breaker reset to CLOSED state")

    def reset_pair(self, pair: str) -> None:
        """Reset per-pair drawdown tracking."""
        if pair in self._pair_states:
            ps = self._pair_states[pair]
            ps.is_paused = False
            ps.paused_until = None
            ps.pause_reason = None
            logger.info(f"Pair {pair} circuit breaker reset")

    def time_until_ready(self) -> Optional[timedelta]:
        """Get time until global circuit breaker is ready."""
        if self._state == CircuitState.CLOSED:
            return None
        if self._state == CircuitState.OPEN and self._cooldown_until:
            remaining = self._cooldown_until - datetime.now(timezone.utc)
            return remaining if remaining.total_seconds() > 0 else None
        return None  # Emergency: no automatic recovery

    # ── Internal ──────────────────────────────────────────────────

    def _global_drawdown_pct(self) -> float:
        """Current global drawdown as % of initial capital."""
        if self._peak_equity <= 0:
            return 0.0
        dd = self._peak_equity - self._current_equity
        return max((dd / self.initial_capital) * 100, 0.0)

    def _trigger_cooldown(self, reason: str) -> None:
        """Trigger global cooldown period."""
        self._state = CircuitState.OPEN
        self._cooldown_until = datetime.now(timezone.utc) + timedelta(
            hours=self.cooldown_hours
        )
        self._trigger_reason = reason
        self._last_state_change = datetime.now(timezone.utc)

        logger.risk_event(
            event_type="circuit_breaker_open",
            details=f"Trading paused until {self._cooldown_until.isoformat()}",
            reason=reason,
            cooldown_hours=self.cooldown_hours,
            global_drawdown=f"{self._global_drawdown_pct():.2f}%",
        )

    def _pause_pair(self, pair: str, reason: str) -> None:
        """Pause a single pair."""
        ps = self._pair_states[pair]
        ps.is_paused = True
        ps.paused_until = datetime.now(timezone.utc) + timedelta(
            hours=self.pair_cooldown_hours
        )
        ps.pause_reason = reason

        logger.risk_event(
            event_type="pair_paused",
            details=f"{pair} paused: {reason}",
            pair=pair,
            cooldown_hours=self.pair_cooldown_hours,
        )

    def _check_cooldown_expired(self) -> None:
        """Check if global cooldown period has expired."""
        if self._state == CircuitState.OPEN and self._cooldown_until:
            if datetime.now(timezone.utc) >= self._cooldown_until:
                logger.info(
                    "Cooldown expired, resetting circuit breaker",
                    cooldown_ended=self._cooldown_until.isoformat(),
                    previous_peak=f"${self._peak_equity:.2f}",
                    current_equity=f"${self._current_equity:.2f}",
                )
                self._state = CircuitState.CLOSED
                self._consecutive_losses = 0
                self._cooldown_until = None
                self._trigger_reason = None
                self._last_state_change = datetime.now(timezone.utc)

                # Reset peak to current equity so drawdown tracking
                # starts fresh.  Without this, the old peak stays and
                # the drawdown threshold is immediately breached again
                # on the next losing trade -- creating an infinite loop
                # of cooldown -> expire -> re-trigger -> cooldown.
                self._peak_equity = self._current_equity

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        pair_states = {}
        for pair, ps in self._pair_states.items():
            pair_states[pair] = {
                "peak_pnl": ps.peak_pnl,
                "current_pnl": ps.current_pnl,
                "is_paused": ps.is_paused,
                "paused_until": ps.paused_until.isoformat() if ps.paused_until else None,
                "pause_reason": ps.pause_reason,
            }

        return {
            "state": self._state.value,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "trigger_reason": self._trigger_reason,
            "last_state_change": self._last_state_change.isoformat(),
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "pair_states": pair_states,
        }

    def from_dict(self, data: dict) -> None:
        """Restore state from dictionary."""
        self._state = CircuitState(data.get("state", "closed"))
        self._consecutive_losses = data.get("consecutive_losses", 0)
        self._trigger_reason = data.get("trigger_reason")
        self._peak_equity = data.get("peak_equity", self.initial_capital)
        self._current_equity = data.get("current_equity", self.initial_capital)

        if data.get("cooldown_until"):
            self._cooldown_until = datetime.fromisoformat(data["cooldown_until"])
        else:
            self._cooldown_until = None

        if data.get("last_state_change"):
            self._last_state_change = datetime.fromisoformat(data["last_state_change"])
        else:
            self._last_state_change = datetime.now(timezone.utc)

        # Restore per-pair states
        for pair, ps_data in data.get("pair_states", {}).items():
            ps = PairDrawdownState(
                peak_pnl=ps_data.get("peak_pnl", 0.0),
                current_pnl=ps_data.get("current_pnl", 0.0),
                is_paused=ps_data.get("is_paused", False),
                pause_reason=ps_data.get("pause_reason"),
            )
            if ps_data.get("paused_until"):
                ps.paused_until = datetime.fromisoformat(ps_data["paused_until"])
            self._pair_states[pair] = ps

        # Check if cooldowns have expired since last save
        self._check_cooldown_expired()
