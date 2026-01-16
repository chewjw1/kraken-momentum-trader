"""
Performance metrics tracking for the trading application.
Tracks P&L, win rate, drawdown, Sharpe ratio, and other key metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import math


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    pair: str
    side: str  # "buy" or "sell"
    entry_price: float
    exit_price: float
    amount: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    fees: float = 0.0


@dataclass
class DailyMetrics:
    """Daily trading metrics."""
    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    volume: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class PerformanceMetrics:
    """Overall performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    current_streak: int = 0  # Positive = wins, negative = losses
    longest_win_streak: int = 0
    longest_loss_streak: int = 0


class MetricsTracker:
    """
    Tracks and calculates trading performance metrics.
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the metrics tracker.

        Args:
            initial_capital: Starting capital for drawdown calculations.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.trades: list[Trade] = []
        self.daily_metrics: dict[str, DailyMetrics] = {}
        self.daily_returns: list[float] = []
        self._current_streak = 0
        self._longest_win_streak = 0
        self._longest_loss_streak = 0

    def record_trade(self, trade: Trade) -> None:
        """
        Record a completed trade and update metrics.

        Args:
            trade: The completed trade to record.
        """
        self.trades.append(trade)

        # Update capital
        self.current_capital += trade.pnl - trade.fees

        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Update daily metrics
        date_str = trade.exit_time.strftime("%Y-%m-%d")
        if date_str not in self.daily_metrics:
            self.daily_metrics[date_str] = DailyMetrics(date=date_str)

        daily = self.daily_metrics[date_str]
        daily.trades += 1
        daily.pnl += trade.pnl - trade.fees
        daily.volume += trade.amount * trade.entry_price

        if trade.pnl > 0:
            daily.wins += 1
        else:
            daily.losses += 1

        # Update streaks
        if trade.pnl > 0:
            if self._current_streak > 0:
                self._current_streak += 1
            else:
                self._current_streak = 1
            self._longest_win_streak = max(self._longest_win_streak, self._current_streak)
        else:
            if self._current_streak < 0:
                self._current_streak -= 1
            else:
                self._current_streak = -1
            self._longest_loss_streak = max(
                self._longest_loss_streak, abs(self._current_streak)
            )

    def record_daily_return(self, return_pct: float) -> None:
        """
        Record a daily return for Sharpe/Sortino calculations.

        Args:
            return_pct: Daily return as a percentage.
        """
        self.daily_returns.append(return_pct)

    def get_metrics(self) -> PerformanceMetrics:
        """
        Calculate and return current performance metrics.

        Returns:
            PerformanceMetrics with current values.
        """
        metrics = PerformanceMetrics()

        if not self.trades:
            return metrics

        # Basic counts
        metrics.total_trades = len(self.trades)
        metrics.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        metrics.losing_trades = sum(1 for t in self.trades if t.pnl <= 0)

        # P&L
        metrics.total_pnl = sum(t.pnl for t in self.trades)
        metrics.total_fees = sum(t.fees for t in self.trades)

        # Win rate
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades * 100

        # Average win/loss
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]

        if wins:
            metrics.average_win = sum(wins) / len(wins)
        if losses:
            metrics.average_loss = sum(losses) / len(losses)

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss

        # Drawdown
        metrics.max_drawdown = self.peak_capital - self.current_capital
        if self.peak_capital > 0:
            metrics.max_drawdown_percent = (
                metrics.max_drawdown / self.peak_capital * 100
            )

        # Sharpe ratio (annualized)
        if len(self.daily_returns) > 1:
            mean_return = sum(self.daily_returns) / len(self.daily_returns)
            variance = sum((r - mean_return) ** 2 for r in self.daily_returns) / (
                len(self.daily_returns) - 1
            )
            std_dev = math.sqrt(variance) if variance > 0 else 0
            if std_dev > 0:
                # Annualize: multiply by sqrt(365) for crypto (trades every day)
                metrics.sharpe_ratio = (mean_return / std_dev) * math.sqrt(365)

        # Sortino ratio (annualized, using only downside deviation)
        if len(self.daily_returns) > 1:
            mean_return = sum(self.daily_returns) / len(self.daily_returns)
            downside_returns = [r for r in self.daily_returns if r < 0]
            if downside_returns:
                downside_variance = sum(r ** 2 for r in downside_returns) / len(
                    downside_returns
                )
                downside_dev = math.sqrt(downside_variance)
                if downside_dev > 0:
                    metrics.sortino_ratio = (mean_return / downside_dev) * math.sqrt(365)

        # Streaks
        metrics.current_streak = self._current_streak
        metrics.longest_win_streak = self._longest_win_streak
        metrics.longest_loss_streak = self._longest_loss_streak

        return metrics

    def get_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """
        Get P&L for a specific date.

        Args:
            date: The date to check. Defaults to today.

        Returns:
            P&L for the date.
        """
        if date is None:
            date = datetime.now(timezone.utc)

        date_str = date.strftime("%Y-%m-%d")
        if date_str in self.daily_metrics:
            return self.daily_metrics[date_str].pnl
        return 0.0

    def get_daily_trade_count(self, date: Optional[datetime] = None) -> int:
        """
        Get trade count for a specific date.

        Args:
            date: The date to check. Defaults to today.

        Returns:
            Number of trades for the date.
        """
        if date is None:
            date = datetime.now(timezone.utc)

        date_str = date.strftime("%Y-%m-%d")
        if date_str in self.daily_metrics:
            return self.daily_metrics[date_str].trades
        return 0

    def get_consecutive_losses(self) -> int:
        """
        Get the current number of consecutive losses.

        Returns:
            Number of consecutive losses (0 if last trade was a win).
        """
        if self._current_streak < 0:
            return abs(self._current_streak)
        return 0

    def is_strategy_validated(
        self,
        min_sharpe: float = 1.0,
        max_drawdown: float = 20.0,
        min_trades: int = 100
    ) -> tuple[bool, str]:
        """
        Check if strategy meets validation criteria for live trading.

        Args:
            min_sharpe: Minimum required Sharpe ratio.
            max_drawdown: Maximum allowed drawdown percentage.
            min_trades: Minimum number of trades required.

        Returns:
            Tuple of (is_valid, reason).
        """
        metrics = self.get_metrics()

        if metrics.total_trades < min_trades:
            return False, f"Insufficient trades: {metrics.total_trades}/{min_trades}"

        if metrics.sharpe_ratio < min_sharpe:
            return False, f"Sharpe ratio too low: {metrics.sharpe_ratio:.2f}/{min_sharpe}"

        if metrics.max_drawdown_percent > max_drawdown:
            return False, f"Drawdown too high: {metrics.max_drawdown_percent:.1f}%/{max_drawdown}%"

        return True, "Strategy validated"

    def to_dict(self) -> dict:
        """
        Serialize metrics to dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        metrics = self.get_metrics()
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": metrics.win_rate,
            "total_pnl": metrics.total_pnl,
            "total_fees": metrics.total_fees,
            "average_win": metrics.average_win,
            "average_loss": metrics.average_loss,
            "profit_factor": metrics.profit_factor,
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_percent": metrics.max_drawdown_percent,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "current_streak": metrics.current_streak,
            "longest_win_streak": metrics.longest_win_streak,
            "longest_loss_streak": metrics.longest_loss_streak,
            "trades_count": len(self.trades),
            "daily_returns_count": len(self.daily_returns),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsTracker":
        """
        Deserialize metrics from dictionary.

        Args:
            data: Dictionary with metrics data.

        Returns:
            MetricsTracker instance.
        """
        tracker = cls(initial_capital=data.get("initial_capital", 10000.0))
        tracker.current_capital = data.get("current_capital", tracker.initial_capital)
        tracker.peak_capital = data.get("peak_capital", tracker.initial_capital)
        tracker._current_streak = data.get("current_streak", 0)
        tracker._longest_win_streak = data.get("longest_win_streak", 0)
        tracker._longest_loss_streak = data.get("longest_loss_streak", 0)
        return tracker
