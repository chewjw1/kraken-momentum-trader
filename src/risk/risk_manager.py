"""
Risk management orchestrator.
Coordinates all risk controls and safeguards.
"""

from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Optional

from ..config.settings import RiskConfig, get_settings
from ..observability.logger import get_logger
from ..observability.metrics import MetricsTracker
from .circuit_breaker import CircuitBreaker, CircuitState
from .position_sizer import PositionSizer

logger = get_logger(__name__)


@dataclass
class RiskCheck:
    """Result of a risk check."""
    allowed: bool
    reason: str
    adjusted_size: Optional[float] = None


@dataclass
class RiskLimits:
    """Current risk limit status."""
    daily_trades_remaining: int
    daily_loss_remaining: float
    max_position_size: float
    current_exposure: float
    max_exposure: float
    exposure_available: float
    circuit_breaker_state: CircuitState


class RiskManager:
    """
    Central risk management coordinator.

    Enforces risk safeguards:
    - Max 40% of capital per trade (configurable)
    - Max 200% total exposure (for Martingale pyramiding)
    - 5 consecutive losses → 2-hour cooldown (circuit breaker)
    - Martingale: max 4 entries, 1.25x multiplier, 5% drop trigger
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        metrics: Optional[MetricsTracker] = None
    ):
        """
        Initialize risk manager.

        Args:
            config: Risk configuration. If None, loaded from settings.
            metrics: Metrics tracker for P&L monitoring.
        """
        if config is None:
            config = get_settings().risk

        self.config = config
        self.metrics = metrics or MetricsTracker()

        # Initialize sub-components
        self.position_sizer = PositionSizer(
            max_position_percent=config.max_position_percent,
            max_position_usd=config.max_position_size_usd
        )

        self.circuit_breaker = CircuitBreaker(
            consecutive_loss_limit=config.circuit_breaker_consecutive_losses,
            cooldown_hours=config.circuit_breaker_cooldown_hours
        )

        # Exposure tracking
        self._current_exposure = 0.0

        # Capital tracking
        self._total_capital = 10000.0  # Will be updated from balances

        # Daily tracking
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_reset_date = date.today()

        logger.info(
            "Risk manager initialized",
            max_position_percent=config.max_position_percent,
            max_exposure_percent=config.max_total_exposure_percent
        )

    def set_capital(self, capital: float) -> None:
        """
        Update total capital for position sizing.

        Args:
            capital: Total account capital.
        """
        self._total_capital = capital
        logger.info(f"Capital updated to ${capital:.2f}")

    def update_exposure(self, exposure: float) -> None:
        """
        Update current exposure.

        Args:
            exposure: Current total exposure in USD.
        """
        self._current_exposure = exposure

    def _check_daily_reset(self) -> None:
        """Reset daily counters if a new day has started."""
        today = date.today()
        if today != self._daily_reset_date:
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._daily_reset_date = today

    def can_trade(self) -> RiskCheck:
        """
        Check if trading is currently allowed.

        Returns:
            RiskCheck indicating if trading is allowed.
        """
        self._check_daily_reset()

        # Check daily trade limit
        if self.config.max_daily_trades > 0 and self._daily_trades >= self.config.max_daily_trades:
            return RiskCheck(
                allowed=False,
                reason=f"Daily trade limit reached: {self._daily_trades}/{self.config.max_daily_trades}"
            )

        # Check daily loss limit
        if self.config.max_daily_loss_percent > 0 and self._daily_pnl < 0:
            max_daily_loss = self._total_capital * (self.config.max_daily_loss_percent / 100)
            if abs(self._daily_pnl) >= max_daily_loss:
                return RiskCheck(
                    allowed=False,
                    reason=f"Daily loss limit reached: ${abs(self._daily_pnl):.2f}/${max_daily_loss:.2f}"
                )

        # Check circuit breaker
        if not self.circuit_breaker.is_trading_allowed():
            state = self.circuit_breaker.get_state()
            return RiskCheck(
                allowed=False,
                reason=f"Circuit breaker active: {state.state.value}"
            )

        # Check exposure limit
        max_exposure = self._total_capital * (self.config.max_total_exposure_percent / 100)
        if self._current_exposure >= max_exposure:
            return RiskCheck(
                allowed=False,
                reason=f"Max exposure reached: ${self._current_exposure:.2f}/${max_exposure:.2f}"
            )

        return RiskCheck(allowed=True, reason="All risk checks passed")

    def check_position_size(
        self,
        requested_size_usd: float,
        pair: str,
        available_balance: Optional[float] = None
    ) -> RiskCheck:
        """
        Check and potentially adjust position size.

        Args:
            requested_size_usd: Requested position size in USD.
            pair: Trading pair.
            available_balance: Available USD balance for trading. If provided,
                               position size will be capped to this amount.

        Returns:
            RiskCheck with adjusted size if needed.
        """
        # First check if trading is allowed
        trade_check = self.can_trade()
        if not trade_check.allowed:
            return trade_check

        # Calculate max position size
        max_size = self.position_sizer.calculate_max_position(self._total_capital)

        # Check remaining exposure
        max_exposure = self._total_capital * (self.config.max_total_exposure_percent / 100)
        remaining_exposure = max_exposure - self._current_exposure

        # Adjust size to fit within limits
        adjusted_size = min(requested_size_usd, max_size, remaining_exposure)

        # Cap by available balance if provided (critical for paper trading)
        if available_balance is not None:
            # Leave 1% buffer for fees
            available_with_buffer = available_balance * 0.99
            if adjusted_size > available_with_buffer:
                logger.risk_event(
                    event_type="position_size_capped_by_balance",
                    details=f"Capped from ${adjusted_size:.2f} to ${available_with_buffer:.2f} (available balance)",
                    pair=pair,
                    available_balance=available_balance
                )
                adjusted_size = available_with_buffer

        if adjusted_size <= 0:
            return RiskCheck(
                allowed=False,
                reason="No capacity for position (insufficient balance or exposure limit reached)",
                adjusted_size=0
            )

        # Minimum position size check
        if adjusted_size < 10:
            return RiskCheck(
                allowed=False,
                reason=f"Position size too small: ${adjusted_size:.2f} (minimum $10)",
                adjusted_size=0
            )

        if adjusted_size < requested_size_usd:
            logger.risk_event(
                event_type="position_size_reduced",
                details=f"Reduced from ${requested_size_usd:.2f} to ${adjusted_size:.2f}",
                pair=pair,
                original_size=requested_size_usd,
                adjusted_size=adjusted_size
            )

        return RiskCheck(
            allowed=True,
            reason="Position size approved",
            adjusted_size=adjusted_size
        )

    def can_add_to_position(
        self,
        additional_size_usd: float,
        current_position_exposure: float,
        num_entries: int,
        pair: str,
        available_balance: Optional[float] = None
    ) -> RiskCheck:
        """
        Check if Martingale add-on is permitted.

        Args:
            additional_size_usd: Size of add-on in USD.
            current_position_exposure: Current position exposure in USD.
            num_entries: Current number of entries in position.
            pair: Trading pair.
            available_balance: Available USD balance for trading. If provided,
                               add-on size will be capped to this amount.

        Returns:
            RiskCheck with adjusted size if needed.
        """
        martingale = self.config.martingale

        # Check if Martingale is enabled
        if not martingale.enabled:
            return RiskCheck(
                allowed=False,
                reason="Martingale is disabled"
            )

        # Check max entries
        if num_entries >= martingale.max_entries:
            return RiskCheck(
                allowed=False,
                reason=f"Max Martingale entries reached: {num_entries}/{martingale.max_entries}"
            )

        # Check circuit breaker
        if not self.circuit_breaker.is_trading_allowed():
            state = self.circuit_breaker.get_state()
            return RiskCheck(
                allowed=False,
                reason=f"Circuit breaker active: {state.state.value}"
            )

        # Start with requested size
        adjusted_size = additional_size_usd

        # Check total exposure after add-on (across ALL positions)
        max_exposure = self._total_capital * (self.config.max_total_exposure_percent / 100)
        new_total_exposure = self._current_exposure + adjusted_size

        if new_total_exposure > max_exposure:
            # Adjust size to fit within exposure limits
            available_exposure = max_exposure - self._current_exposure
            if available_exposure <= 0:
                return RiskCheck(
                    allowed=False,
                    reason=f"Max exposure reached: ${current_position_exposure:.2f}/${max_exposure:.2f}"
                )
            adjusted_size = available_exposure

        # Cap by available balance if provided (critical for paper trading)
        if available_balance is not None:
            # Leave 1% buffer for fees
            available_with_buffer = available_balance * 0.99
            if adjusted_size > available_with_buffer:
                if available_with_buffer <= 0:
                    return RiskCheck(
                        allowed=False,
                        reason=f"Insufficient balance for Martingale add-on (available: ${available_balance:.2f})"
                    )
                adjusted_size = available_with_buffer

        # Minimum position size check
        if adjusted_size < 10:
            return RiskCheck(
                allowed=False,
                reason=f"Martingale add-on size too small: ${adjusted_size:.2f} (minimum $10)"
            )

        if adjusted_size < additional_size_usd:
            logger.risk_event(
                event_type="martingale_size_reduced",
                details=f"Reduced from ${additional_size_usd:.2f} to ${adjusted_size:.2f}",
                pair=pair
            )
            return RiskCheck(
                allowed=True,
                reason="Martingale add-on approved (size adjusted)",
                adjusted_size=adjusted_size
            )

        return RiskCheck(
            allowed=True,
            reason="Martingale add-on approved",
            adjusted_size=adjusted_size
        )

    def calculate_martingale_size(
        self,
        last_entry_size_usd: float
    ) -> float:
        """
        Calculate the size for next Martingale entry.

        Args:
            last_entry_size_usd: Size of the most recent entry.

        Returns:
            Size for next entry in USD.
        """
        multiplier = self.config.martingale.size_multiplier
        return last_entry_size_usd * multiplier

    def record_trade_result(self, pnl: float, pair: str) -> None:
        """
        Record trade result for risk tracking.

        Args:
            pnl: Trade profit/loss.
            pair: Trading pair.
        """
        self._check_daily_reset()

        # Update daily counters
        self._daily_trades += 1
        self._daily_pnl += pnl

        if pnl < 0:
            self.circuit_breaker.record_loss()
            logger.risk_event(
                event_type="loss_recorded",
                details=f"Loss of ${abs(pnl):.2f} on {pair}",
                consecutive_losses=self.circuit_breaker.get_state().consecutive_losses
            )
        else:
            self.circuit_breaker.record_win()

        # Update metrics
        self.metrics.record_daily_return(pnl / self._total_capital * 100)

    def should_emergency_exit(self) -> tuple[bool, str]:
        """
        Check if all positions should be closed immediately.

        Note: Emergency exit is now only triggered by manual circuit breaker
        activation, not by drawdown limits.

        Returns:
            Tuple of (should_exit, reason).
        """
        state = self.circuit_breaker.get_state()
        if state.state == CircuitState.EMERGENCY:
            return True, state.trigger_reason or "Emergency stop activated"

        return False, ""

    def get_limits(self) -> RiskLimits:
        """
        Get current risk limit status.

        Returns:
            RiskLimits with current status.
        """
        self._check_daily_reset()
        max_exposure = self._total_capital * (self.config.max_total_exposure_percent / 100)

        # Daily trade remaining
        if self.config.max_daily_trades > 0:
            daily_trades_remaining = max(0, self.config.max_daily_trades - self._daily_trades)
        else:
            daily_trades_remaining = -1

        # Daily loss remaining
        if self.config.max_daily_loss_percent > 0:
            max_daily_loss = self._total_capital * (self.config.max_daily_loss_percent / 100)
            current_loss = abs(self._daily_pnl) if self._daily_pnl < 0 else 0.0
            daily_loss_remaining = max(0.0, max_daily_loss - current_loss)
        else:
            daily_loss_remaining = -1

        return RiskLimits(
            daily_trades_remaining=daily_trades_remaining,
            daily_loss_remaining=daily_loss_remaining,
            max_position_size=self.position_sizer.calculate_max_position(self._total_capital),
            current_exposure=self._current_exposure,
            max_exposure=max_exposure,
            exposure_available=max(0, max_exposure - self._current_exposure),
            circuit_breaker_state=self.circuit_breaker.get_state().state
        )

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (use with caution)."""
        logger.warning("Manual circuit breaker reset requested")
        self.circuit_breaker.reset()

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            "total_capital": self._total_capital,
            "current_exposure": self._current_exposure,
            "circuit_breaker": self.circuit_breaker.to_dict(),
            "daily_trades": self._daily_trades,
            "daily_pnl": self._daily_pnl,
            "daily_reset_date": self._daily_reset_date.isoformat(),
        }

    def from_dict(self, data: dict) -> None:
        """Restore state from dictionary."""
        self._total_capital = data.get("total_capital", 10000.0)
        self._current_exposure = data.get("current_exposure", 0.0)
        self._daily_trades = data.get("daily_trades", 0)
        self._daily_pnl = data.get("daily_pnl", 0.0)
        if data.get("daily_reset_date"):
            self._daily_reset_date = date.fromisoformat(data["daily_reset_date"])
        else:
            self._daily_reset_date = date.today()

        if "circuit_breaker" in data:
            self.circuit_breaker.from_dict(data["circuit_breaker"])
