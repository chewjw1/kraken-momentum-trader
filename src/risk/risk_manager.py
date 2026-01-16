"""
Risk management orchestrator.
Coordinates all risk controls and safeguards.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
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

    Enforces all non-negotiable risk safeguards:
    - Max 5% of capital per trade
    - Max 20% total exposure
    - 3% daily loss limit → pause trading
    - 3 consecutive losses → 4-hour cooldown
    - 10% drawdown → emergency exit all positions
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
            cooldown_hours=config.circuit_breaker_cooldown_hours,
            max_drawdown_percent=config.max_drawdown_percent
        )

        # Daily tracking
        self._daily_trade_count = 0
        self._daily_loss = 0.0
        self._current_date: Optional[str] = None
        self._current_exposure = 0.0

        # Capital tracking
        self._total_capital = 10000.0  # Will be updated from balances

        logger.info(
            "Risk manager initialized",
            max_position_percent=config.max_position_percent,
            max_exposure_percent=config.max_total_exposure_percent,
            daily_loss_limit=config.max_daily_loss_percent
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

    def can_trade(self) -> RiskCheck:
        """
        Check if trading is currently allowed.

        Returns:
            RiskCheck indicating if trading is allowed.
        """
        self._check_date_reset()

        # Check circuit breaker
        if not self.circuit_breaker.is_trading_allowed():
            state = self.circuit_breaker.get_state()
            return RiskCheck(
                allowed=False,
                reason=f"Circuit breaker active: {state.state.value}"
            )

        # Check daily trade limit
        if self._daily_trade_count >= self.config.max_daily_trades:
            return RiskCheck(
                allowed=False,
                reason=f"Daily trade limit reached: {self._daily_trade_count}/{self.config.max_daily_trades}"
            )

        # Check daily loss limit
        max_daily_loss = self._total_capital * (self.config.max_daily_loss_percent / 100)
        if self._daily_loss >= max_daily_loss:
            return RiskCheck(
                allowed=False,
                reason=f"Daily loss limit reached: ${self._daily_loss:.2f}/${max_daily_loss:.2f}"
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
        pair: str
    ) -> RiskCheck:
        """
        Check and potentially adjust position size.

        Args:
            requested_size_usd: Requested position size in USD.
            pair: Trading pair.

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

        if adjusted_size <= 0:
            return RiskCheck(
                allowed=False,
                reason="No capacity for position (exposure limit reached)",
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

    def record_trade_result(self, pnl: float, pair: str) -> None:
        """
        Record trade result for risk tracking.

        Args:
            pnl: Trade profit/loss.
            pair: Trading pair.
        """
        self._check_date_reset()
        self._daily_trade_count += 1

        if pnl < 0:
            self._daily_loss += abs(pnl)
            self.circuit_breaker.record_loss()
            logger.risk_event(
                event_type="loss_recorded",
                details=f"Loss of ${abs(pnl):.2f} on {pair}",
                daily_loss=self._daily_loss,
                consecutive_losses=self.circuit_breaker.get_state().consecutive_losses
            )
        else:
            self.circuit_breaker.record_win()

        # Update metrics
        self.metrics.record_daily_return(pnl / self._total_capital * 100)

        # Check for emergency stop
        metrics = self.metrics.get_metrics()
        if metrics.max_drawdown_percent >= self.config.max_drawdown_percent:
            self.circuit_breaker.trigger_emergency_stop(
                f"Max drawdown {metrics.max_drawdown_percent:.1f}% exceeded limit"
            )
            logger.risk_event(
                event_type="emergency_stop",
                details=f"Drawdown of {metrics.max_drawdown_percent:.1f}% exceeded {self.config.max_drawdown_percent}%"
            )

    def should_emergency_exit(self) -> tuple[bool, str]:
        """
        Check if all positions should be closed immediately.

        Returns:
            Tuple of (should_exit, reason).
        """
        state = self.circuit_breaker.get_state()
        if state.state == CircuitState.EMERGENCY:
            return True, state.trigger_reason or "Emergency stop activated"

        metrics = self.metrics.get_metrics()
        if metrics.max_drawdown_percent >= self.config.max_drawdown_percent:
            return True, f"Drawdown {metrics.max_drawdown_percent:.1f}% exceeded limit"

        return False, ""

    def get_limits(self) -> RiskLimits:
        """
        Get current risk limit status.

        Returns:
            RiskLimits with current status.
        """
        self._check_date_reset()

        max_daily_loss = self._total_capital * (self.config.max_daily_loss_percent / 100)
        max_exposure = self._total_capital * (self.config.max_total_exposure_percent / 100)

        return RiskLimits(
            daily_trades_remaining=max(0, self.config.max_daily_trades - self._daily_trade_count),
            daily_loss_remaining=max(0, max_daily_loss - self._daily_loss),
            max_position_size=self.position_sizer.calculate_max_position(self._total_capital),
            current_exposure=self._current_exposure,
            max_exposure=max_exposure,
            exposure_available=max(0, max_exposure - self._current_exposure),
            circuit_breaker_state=self.circuit_breaker.get_state().state
        )

    def _check_date_reset(self) -> None:
        """Reset daily counters if date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._current_date != today:
            if self._current_date is not None:
                logger.info(
                    f"New trading day: resetting daily counters",
                    previous_trades=self._daily_trade_count,
                    previous_loss=self._daily_loss
                )
            self._current_date = today
            self._daily_trade_count = 0
            self._daily_loss = 0.0

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (use with caution)."""
        logger.warning("Manual circuit breaker reset requested")
        self.circuit_breaker.reset()

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            "total_capital": self._total_capital,
            "current_exposure": self._current_exposure,
            "daily_trade_count": self._daily_trade_count,
            "daily_loss": self._daily_loss,
            "current_date": self._current_date,
            "circuit_breaker": self.circuit_breaker.to_dict(),
        }

    def from_dict(self, data: dict) -> None:
        """Restore state from dictionary."""
        self._total_capital = data.get("total_capital", 10000.0)
        self._current_exposure = data.get("current_exposure", 0.0)
        self._daily_trade_count = data.get("daily_trade_count", 0)
        self._daily_loss = data.get("daily_loss", 0.0)
        self._current_date = data.get("current_date")

        if "circuit_breaker" in data:
            self.circuit_breaker.from_dict(data["circuit_breaker"])
