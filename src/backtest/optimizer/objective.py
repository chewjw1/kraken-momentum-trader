"""
Objective functions for parameter optimization.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import math


@dataclass
class BacktestMetrics:
    """Metrics from a backtest run."""
    total_return: float  # Percentage return (e.g., 0.15 = 15%)
    sharpe_ratio: float  # Risk-adjusted return
    max_drawdown_percent: float  # Maximum drawdown percentage
    profit_factor: float  # Gross profit / Gross loss
    win_rate: float  # Percentage of winning trades
    total_trades: int  # Number of completed trades
    total_pnl: float  # Absolute P&L in USD
    total_fees: float  # Total fees paid


def composite_score(metrics: BacktestMetrics) -> float:
    """
    Calculate composite optimization score.

    Optimized for sideways markets:
    - Rewards consistent trading activity (more trades = more opportunities)
    - Values win rate and profit factor over large single returns
    - Penalizes extreme drawdowns but tolerates small negative returns
    - Rewards capital preservation with active trading

    Args:
        metrics: BacktestMetrics from a backtest run.

    Returns:
        Composite score (higher is better).
    """
    # Base score components
    return_score = metrics.total_return * 100  # Convert to percentage points

    # Sharpe ratio contribution (scaled)
    sharpe_score = metrics.sharpe_ratio * 10

    # Profit factor contribution (scaled, capped at 3)
    pf_score = min(metrics.profit_factor, 3.0) * 5

    # Win rate contribution (scaled)
    win_rate_score = metrics.win_rate * 20  # 0-20 points based on win rate

    # Drawdown penalty
    drawdown_penalty = metrics.max_drawdown_percent

    # Trade activity score - reward more trades in sideways markets
    # Sigmoid-like scaling: rapidly increases up to ~50 trades, then flattens
    trade_activity_score = min(metrics.total_trades / 10, 5.0) * 3  # Up to 15 points

    # Weighted composite - prioritizing activity and consistency
    score = (
        return_score * 0.25 +       # Reduced from 0.4
        sharpe_score * 0.15 +       # Reduced from 0.3
        pf_score * 0.20 +           # Same
        win_rate_score * 0.20 +     # New - reward winning
        trade_activity_score * 0.20 -  # New - reward activity
        drawdown_penalty * 0.10
    )

    # Severe penalty for very few trades (strategy not engaging with market)
    if metrics.total_trades < 10:
        trade_penalty = (10 - metrics.total_trades) * 5  # Harsh penalty
        score -= trade_penalty
    elif metrics.total_trades < 20:
        trade_penalty = (20 - metrics.total_trades) * 1  # Mild penalty
        score -= trade_penalty

    # Moderate penalty for negative returns (but don't destroy score)
    if metrics.total_return < 0:
        # Scale penalty by magnitude - small losses are OK in sideways markets
        loss_magnitude = abs(metrics.total_return)
        if loss_magnitude < 0.05:  # Less than 5% loss
            score -= loss_magnitude * 20  # Mild penalty
        elif loss_magnitude < 0.15:  # 5-15% loss
            score -= loss_magnitude * 40  # Moderate penalty
        else:  # > 15% loss
            score -= loss_magnitude * 60  # Severe penalty

    # Severe penalty for extreme drawdown
    if metrics.max_drawdown_percent > 30:
        score -= (metrics.max_drawdown_percent - 30) * 2

    # Bonus for good win rate with sufficient trades (sideways market success)
    if metrics.total_trades >= 15 and metrics.win_rate > 0.5:
        score += (metrics.win_rate - 0.5) * 15

    # Bonus for high trade count with positive returns (ideal outcome)
    if metrics.total_trades >= 30 and metrics.total_return > 0:
        score += 10  # Flat bonus for active profitable trading

    return score


def sharpe_score(metrics: BacktestMetrics) -> float:
    """
    Optimization score based primarily on Sharpe ratio.

    Args:
        metrics: BacktestMetrics from a backtest run.

    Returns:
        Score based on Sharpe ratio (higher is better).
    """
    score = metrics.sharpe_ratio

    # Penalty for too few trades
    if metrics.total_trades < 20:
        score -= (20 - metrics.total_trades) * 0.1

    # Penalty for extreme drawdown
    if metrics.max_drawdown_percent > 30:
        score -= (metrics.max_drawdown_percent - 30) * 0.05

    return score


def return_score(metrics: BacktestMetrics) -> float:
    """
    Optimization score based primarily on total return.

    Args:
        metrics: BacktestMetrics from a backtest run.

    Returns:
        Score based on return (higher is better).
    """
    score = metrics.total_return * 100

    # Penalty for too few trades
    if metrics.total_trades < 20:
        score -= (20 - metrics.total_trades) * 1

    # Penalty for extreme drawdown relative to return
    if metrics.max_drawdown_percent > 0:
        risk_reward = metrics.total_return * 100 / metrics.max_drawdown_percent
        if risk_reward < 0.5:
            score *= 0.8  # 20% penalty for poor risk/reward

    return score


def profit_factor_score(metrics: BacktestMetrics) -> float:
    """
    Optimization score based primarily on profit factor.

    Args:
        metrics: BacktestMetrics from a backtest run.

    Returns:
        Score based on profit factor (higher is better).
    """
    # Cap profit factor to avoid overfitting to lucky trades
    score = min(metrics.profit_factor, 5.0) * 10

    # Penalty for too few trades
    if metrics.total_trades < 20:
        score -= (20 - metrics.total_trades) * 0.5

    # Bonus for return > 0
    if metrics.total_return > 0:
        score += metrics.total_return * 10

    return score


class ObjectiveFunction:
    """
    Objective function for Optuna optimization.

    Evaluates parameter sets by running backtests and computing scores.
    """

    SCORE_FUNCTIONS = {
        "composite": composite_score,
        "sharpe": sharpe_score,
        "return": return_score,
        "profit_factor": profit_factor_score,
    }

    def __init__(
        self,
        objective_type: str = "composite",
        min_trades: int = 20
    ):
        """
        Initialize objective function.

        Args:
            objective_type: Type of objective ("composite", "sharpe", "return", "profit_factor").
            min_trades: Minimum trades required to avoid heavy penalty.
        """
        if objective_type not in self.SCORE_FUNCTIONS:
            raise ValueError(
                f"Unknown objective type: {objective_type}. "
                f"Valid options: {list(self.SCORE_FUNCTIONS.keys())}"
            )

        self.objective_type = objective_type
        self.score_fn = self.SCORE_FUNCTIONS[objective_type]
        self.min_trades = min_trades

    def calculate_score(self, metrics: BacktestMetrics) -> float:
        """
        Calculate optimization score for given metrics.

        Args:
            metrics: BacktestMetrics from backtest run.

        Returns:
            Optimization score (higher is better).
        """
        return self.score_fn(metrics)

    def aggregate_multi_pair_scores(
        self,
        pair_metrics: Dict[str, BacktestMetrics]
    ) -> float:
        """
        Aggregate scores from multiple pairs.

        Uses weighted average with emphasis on consistency.

        Args:
            pair_metrics: Dictionary mapping pair names to their metrics.

        Returns:
            Aggregated score.
        """
        if not pair_metrics:
            return float('-inf')

        scores = []
        weights = []

        for pair, metrics in pair_metrics.items():
            score = self.calculate_score(metrics)
            scores.append(score)

            # Weight by trade count (more trades = more reliable)
            weight = min(metrics.total_trades / self.min_trades, 2.0)
            weights.append(weight)

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return float('-inf')

        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Consistency bonus: if all pairs are profitable, add bonus
        all_profitable = all(m.total_return > 0 for m in pair_metrics.values())
        if all_profitable:
            weighted_avg *= 1.1  # 10% bonus

        # Consistency penalty: high variance across pairs indicates overfitting
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = math.sqrt(variance)

            # Penalize high variance (relative to mean)
            if mean_score > 0:
                cv = std_dev / mean_score  # Coefficient of variation
                if cv > 0.5:  # High variance
                    weighted_avg *= (1.0 - min(cv - 0.5, 0.3))  # Up to 30% penalty

        return weighted_avg

    @staticmethod
    def from_backtest_results(results: Any, initial_capital: float = 10000.0) -> BacktestMetrics:
        """
        Convert BacktestResults to BacktestMetrics.

        Args:
            results: BacktestResults object from backtest engine.
            initial_capital: Initial capital used in backtest.

        Returns:
            BacktestMetrics object.
        """
        metrics = results.metrics

        # Calculate total return from total_pnl and initial capital
        net_pnl = metrics.total_pnl - metrics.total_fees
        total_return = net_pnl / initial_capital if initial_capital > 0 else 0.0

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=metrics.sharpe_ratio if hasattr(metrics, 'sharpe_ratio') else 0.0,
            max_drawdown_percent=metrics.max_drawdown_percent,
            profit_factor=metrics.profit_factor,
            win_rate=metrics.win_rate,
            total_trades=metrics.total_trades,
            total_pnl=metrics.total_pnl,
            total_fees=metrics.total_fees,
        )
