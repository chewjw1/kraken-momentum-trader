"""
Adaptive Pair Manager for dynamic pair selection.

Tracks real-time performance per trading pair and automatically:
- Disables underperforming pairs
- Re-enables pairs when conditions improve
- Adjusts position sizing based on confidence

Uses rolling window metrics to adapt to changing market conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeRecord:
    """Record of a single trade for performance tracking."""
    pair: str
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    is_win: bool


@dataclass
class PairPerformance:
    """Rolling performance metrics for a trading pair."""
    pair: str
    trades: deque = field(default_factory=lambda: deque(maxlen=50))
    total_trades: int = 0
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    is_enabled: bool = True
    disabled_at: Optional[datetime] = None
    disabled_reason: Optional[str] = None
    cooldown_until: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate from recent trades."""
        if len(self.trades) == 0:
            return 0.5  # Neutral when no data
        wins = sum(1 for t in self.trades if t.is_win)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
        return gross_profit / gross_loss

    @property
    def avg_pnl_percent(self) -> float:
        """Calculate average P&L percentage."""
        if len(self.trades) == 0:
            return 0.0
        return sum(t.pnl_percent for t in self.trades) / len(self.trades)

    @property
    def recent_pnl(self) -> float:
        """Calculate P&L from last 10 trades."""
        recent = list(self.trades)[-10:]
        return sum(t.pnl for t in recent)

    @property
    def sharpe_estimate(self) -> float:
        """Estimate Sharpe-like ratio from recent trades."""
        if len(self.trades) < 5:
            return 0.0
        returns = [t.pnl_percent for t in self.trades]
        avg_return = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0
        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5
        if std_dev == 0:
            return 0.0
        return avg_return / std_dev


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive pair management."""
    # Performance thresholds for disabling
    min_win_rate: float = 0.35  # Disable if win rate drops below 35%
    min_profit_factor: float = 0.7  # Disable if profit factor drops below 0.7
    max_consecutive_losses: int = 5  # Disable after 5 consecutive losses
    min_trades_for_decision: int = 10  # Need 10 trades before making decisions

    # Re-enable conditions
    cooldown_hours: float = 2.0  # Minimum hours before re-enabling
    reenable_win_rate: float = 0.45  # Re-enable if simulated win rate > 45%

    # Position sizing adjustments
    confidence_scaling: bool = True  # Scale position size by confidence
    min_position_scale: float = 0.5  # Minimum position size multiplier
    max_position_scale: float = 1.5  # Maximum position size multiplier


class AdaptivePairManager:
    """
    Manages trading pairs dynamically based on performance.

    Features:
    - Tracks performance per pair with rolling window
    - Auto-disables underperforming pairs
    - Re-enables pairs after cooldown if conditions improve
    - Provides position size scaling based on confidence
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize the adaptive pair manager.

        Args:
            config: Configuration for adaptive behavior.
        """
        self.config = config or AdaptiveConfig()
        self._pairs: Dict[str, PairPerformance] = {}
        self._global_stats = {
            "total_trades": 0,
            "total_pnl": 0.0,
            "pairs_disabled": 0,
            "pairs_reenabled": 0,
        }

        logger.info(
            "Adaptive pair manager initialized",
            min_win_rate=self.config.min_win_rate,
            max_consecutive_losses=self.config.max_consecutive_losses
        )

    def register_pair(self, pair: str) -> None:
        """
        Register a trading pair for tracking.

        Args:
            pair: Trading pair symbol.
        """
        if pair not in self._pairs:
            self._pairs[pair] = PairPerformance(pair=pair)
            logger.info(f"Registered pair for adaptive tracking: {pair}")

    def record_trade(
        self,
        pair: str,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        pnl_percent: float
    ) -> None:
        """
        Record a completed trade for performance tracking.

        Args:
            pair: Trading pair.
            entry_time: Trade entry time.
            exit_time: Trade exit time.
            pnl: Profit/loss in USD.
            pnl_percent: Profit/loss percentage.
        """
        if pair not in self._pairs:
            self.register_pair(pair)

        perf = self._pairs[pair]
        is_win = pnl > 0

        trade = TradeRecord(
            pair=pair,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_percent=pnl_percent,
            is_win=is_win
        )

        perf.trades.append(trade)
        perf.total_trades += 1
        perf.total_pnl += pnl

        if is_win:
            perf.wins += 1
            perf.consecutive_losses = 0
        else:
            perf.losses += 1
            perf.consecutive_losses += 1
            perf.max_consecutive_losses = max(
                perf.max_consecutive_losses,
                perf.consecutive_losses
            )

        self._global_stats["total_trades"] += 1
        self._global_stats["total_pnl"] += pnl

        logger.info(
            f"Trade recorded for {pair}",
            pnl=pnl,
            pnl_percent=pnl_percent,
            win_rate=perf.win_rate,
            consecutive_losses=perf.consecutive_losses
        )

        # Check if pair should be disabled
        self._evaluate_pair_status(pair)

    def _evaluate_pair_status(self, pair: str) -> None:
        """Evaluate if a pair should be disabled or re-enabled."""
        perf = self._pairs[pair]
        now = datetime.now(timezone.utc)

        # Not enough data yet
        if len(perf.trades) < self.config.min_trades_for_decision:
            return

        if perf.is_enabled:
            # Check disable conditions
            disable_reason = None

            if perf.win_rate < self.config.min_win_rate:
                disable_reason = f"Win rate {perf.win_rate:.1%} < {self.config.min_win_rate:.1%}"

            elif perf.profit_factor < self.config.min_profit_factor:
                disable_reason = f"Profit factor {perf.profit_factor:.2f} < {self.config.min_profit_factor}"

            elif perf.consecutive_losses >= self.config.max_consecutive_losses:
                disable_reason = f"Consecutive losses: {perf.consecutive_losses}"

            if disable_reason:
                self._disable_pair(pair, disable_reason)

        else:
            # Check re-enable conditions
            if perf.cooldown_until and now < perf.cooldown_until:
                return  # Still in cooldown

            # Check if performance has improved
            if perf.win_rate >= self.config.reenable_win_rate:
                self._enable_pair(pair)

    def _disable_pair(self, pair: str, reason: str) -> None:
        """Disable a trading pair."""
        perf = self._pairs[pair]
        now = datetime.now(timezone.utc)

        perf.is_enabled = False
        perf.disabled_at = now
        perf.disabled_reason = reason
        perf.cooldown_until = now + timedelta(hours=self.config.cooldown_hours)

        self._global_stats["pairs_disabled"] += 1

        logger.warning(
            f"PAIR DISABLED: {pair}",
            reason=reason,
            win_rate=perf.win_rate,
            profit_factor=perf.profit_factor,
            cooldown_until=perf.cooldown_until.isoformat()
        )

    def _enable_pair(self, pair: str) -> None:
        """Re-enable a trading pair."""
        perf = self._pairs[pair]

        perf.is_enabled = True
        perf.disabled_at = None
        perf.disabled_reason = None
        perf.cooldown_until = None
        perf.consecutive_losses = 0  # Reset consecutive losses

        self._global_stats["pairs_reenabled"] += 1

        logger.info(
            f"PAIR RE-ENABLED: {pair}",
            win_rate=perf.win_rate,
            profit_factor=perf.profit_factor
        )

    def is_pair_enabled(self, pair: str) -> bool:
        """
        Check if a pair is currently enabled for trading.

        Args:
            pair: Trading pair.

        Returns:
            True if pair is enabled.
        """
        if pair not in self._pairs:
            return True  # New pairs are enabled by default

        return self._pairs[pair].is_enabled

    def get_position_scale(self, pair: str) -> float:
        """
        Get position size scaling factor based on performance.

        Args:
            pair: Trading pair.

        Returns:
            Multiplier for position size (0.5 - 1.5).
        """
        if not self.config.confidence_scaling:
            return 1.0

        if pair not in self._pairs:
            return 1.0

        perf = self._pairs[pair]

        if len(perf.trades) < 5:
            return 1.0  # Not enough data

        # Scale based on win rate and profit factor
        win_rate_score = (perf.win_rate - 0.4) / 0.3  # 0.4-0.7 maps to 0-1
        pf_score = (min(perf.profit_factor, 2.0) - 0.5) / 1.5  # 0.5-2.0 maps to 0-1

        # Average the scores
        confidence = (win_rate_score + pf_score) / 2

        # Clamp and scale
        confidence = max(0, min(1, confidence))
        scale = self.config.min_position_scale + (
            confidence * (self.config.max_position_scale - self.config.min_position_scale)
        )

        return scale

    def get_enabled_pairs(self, pairs: List[str]) -> List[str]:
        """
        Filter a list of pairs to only enabled ones.

        Args:
            pairs: List of trading pairs.

        Returns:
            List of enabled pairs.
        """
        return [p for p in pairs if self.is_pair_enabled(p)]

    def get_pair_status(self, pair: str) -> dict:
        """
        Get detailed status for a pair.

        Args:
            pair: Trading pair.

        Returns:
            Dictionary with pair status and metrics.
        """
        if pair not in self._pairs:
            return {
                "pair": pair,
                "registered": False,
                "enabled": True,
                "trades": 0,
            }

        perf = self._pairs[pair]
        return {
            "pair": pair,
            "registered": True,
            "enabled": perf.is_enabled,
            "trades": len(perf.trades),
            "total_trades": perf.total_trades,
            "win_rate": perf.win_rate,
            "profit_factor": perf.profit_factor,
            "avg_pnl_percent": perf.avg_pnl_percent,
            "total_pnl": perf.total_pnl,
            "consecutive_losses": perf.consecutive_losses,
            "position_scale": self.get_position_scale(pair),
            "disabled_reason": perf.disabled_reason,
            "cooldown_until": perf.cooldown_until.isoformat() if perf.cooldown_until else None,
        }

    def get_summary(self) -> dict:
        """Get summary of all pairs."""
        enabled = [p for p, perf in self._pairs.items() if perf.is_enabled]
        disabled = [p for p, perf in self._pairs.items() if not perf.is_enabled]

        return {
            "total_pairs": len(self._pairs),
            "enabled_pairs": enabled,
            "disabled_pairs": disabled,
            "global_stats": self._global_stats,
            "pair_details": {
                pair: self.get_pair_status(pair)
                for pair in self._pairs
            }
        }

    def force_enable(self, pair: str) -> None:
        """Force enable a pair (manual override)."""
        if pair in self._pairs:
            self._enable_pair(pair)
            logger.info(f"Manually force-enabled pair: {pair}")

    def force_disable(self, pair: str, reason: str = "Manual disable") -> None:
        """Force disable a pair (manual override)."""
        if pair not in self._pairs:
            self.register_pair(pair)
        self._disable_pair(pair, reason)
        logger.info(f"Manually force-disabled pair: {pair}")

    def to_dict(self) -> dict:
        """Serialize state to dictionary for persistence."""
        return {
            "global_stats": self._global_stats,
            "pairs": {
                pair: {
                    "is_enabled": perf.is_enabled,
                    "total_trades": perf.total_trades,
                    "total_pnl": perf.total_pnl,
                    "wins": perf.wins,
                    "losses": perf.losses,
                    "consecutive_losses": perf.consecutive_losses,
                    "disabled_reason": perf.disabled_reason,
                    "cooldown_until": perf.cooldown_until.isoformat() if perf.cooldown_until else None,
                }
                for pair, perf in self._pairs.items()
            }
        }

    def from_dict(self, data: dict) -> None:
        """Restore state from dictionary."""
        self._global_stats = data.get("global_stats", self._global_stats)

        for pair, pair_data in data.get("pairs", {}).items():
            if pair not in self._pairs:
                self.register_pair(pair)

            perf = self._pairs[pair]
            perf.is_enabled = pair_data.get("is_enabled", True)
            perf.total_trades = pair_data.get("total_trades", 0)
            perf.total_pnl = pair_data.get("total_pnl", 0.0)
            perf.wins = pair_data.get("wins", 0)
            perf.losses = pair_data.get("losses", 0)
            perf.consecutive_losses = pair_data.get("consecutive_losses", 0)
            perf.disabled_reason = pair_data.get("disabled_reason")

            cooldown_str = pair_data.get("cooldown_until")
            if cooldown_str:
                perf.cooldown_until = datetime.fromisoformat(cooldown_str)
