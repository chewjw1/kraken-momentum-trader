"""
Results reporter for backtest output and export.
"""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from .engine import BacktestResults
from .position import CompletedTrade


@dataclass
class PeriodStats:
    """Statistics for a time period."""
    period: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    is_accretive: bool


class ResultsReporter:
    """
    Generates reports and exports for backtest results.

    Features:
    - Console summary output
    - Period breakdown (monthly/weekly)
    - CSV/JSON trade export
    """

    def __init__(self, results: BacktestResults):
        """
        Initialize the reporter.

        Args:
            results: BacktestResults from the engine.
        """
        self.results = results

    def print_summary(self) -> None:
        """Print a comprehensive summary to console."""
        config = self.results.config
        metrics = self.results.metrics

        # Header
        print("=" * 80)
        print(f"{'BACKTEST RESULTS: ' + config.pair:^80}")
        print("=" * 80)

        # Period info
        period_days = config.period_days
        print(f"Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')} ({period_days} days)")
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Interval: {config.interval} minutes")

        # Performance summary
        print()
        print("-" * 80)
        print(f"{'PERFORMANCE SUMMARY':^80}")
        print("-" * 80)

        final_capital = config.initial_capital + metrics.total_pnl - metrics.total_fees
        total_return = ((final_capital - config.initial_capital) / config.initial_capital) * 100

        print(f"{'Total Trades:':<30} {metrics.total_trades}")
        print(f"{'Winning Trades:':<30} {metrics.winning_trades}")
        print(f"{'Losing Trades:':<30} {metrics.losing_trades}")
        print(f"{'Win Rate:':<30} {metrics.win_rate:.1f}%")
        print()
        print(f"{'Gross P&L:':<30} ${metrics.total_pnl:,.2f}")
        print(f"{'Total Fees:':<30} ${metrics.total_fees:,.2f}")
        print(f"{'Net P&L:':<30} ${metrics.total_pnl - metrics.total_fees:,.2f} ({total_return:+.2f}%)")
        print(f"{'Final Capital:':<30} ${final_capital:,.2f}")

        # Risk metrics
        print()
        print("-" * 80)
        print(f"{'RISK METRICS':^80}")
        print("-" * 80)

        print(f"{'Sharpe Ratio:':<30} {metrics.sharpe_ratio:.2f}")
        print(f"{'Sortino Ratio:':<30} {metrics.sortino_ratio:.2f}")
        print(f"{'Max Drawdown:':<30} ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.1f}%)")
        print(f"{'Profit Factor:':<30} {metrics.profit_factor:.2f}")

        if metrics.average_win != 0 or metrics.average_loss != 0:
            print()
            print(f"{'Average Win:':<30} ${metrics.average_win:,.2f}")
            print(f"{'Average Loss:':<30} ${metrics.average_loss:,.2f}")

        print(f"{'Longest Win Streak:':<30} {metrics.longest_win_streak}")
        print(f"{'Longest Loss Streak:':<30} {metrics.longest_loss_streak}")

        # Martingale stats
        martingale_trades = [t for t in self.results.trades if t.num_entries > 1]
        if martingale_trades:
            print()
            print("-" * 80)
            print(f"{'MARTINGALE STATISTICS':^80}")
            print("-" * 80)

            avg_entries = sum(t.num_entries for t in martingale_trades) / len(martingale_trades)
            max_entries = max(t.num_entries for t in martingale_trades)
            martingale_wins = sum(1 for t in martingale_trades if t.is_winner)

            print(f"{'Trades with Add-ons:':<30} {len(martingale_trades)}")
            print(f"{'Avg Entries per Trade:':<30} {avg_entries:.1f}")
            print(f"{'Max Entries in Trade:':<30} {max_entries}")
            print(f"{'Martingale Win Rate:':<30} {martingale_wins / len(martingale_trades) * 100:.1f}%")

        # Period breakdown
        self._print_period_breakdown()

        print("=" * 80)

    def _print_period_breakdown(self, period: str = "monthly") -> None:
        """Print period-by-period breakdown."""
        stats = self.get_period_breakdown(period)

        if not stats:
            return

        print()
        print("-" * 80)
        print(f"{'PERIOD BREAKDOWN (' + period.upper() + ')':^80}")
        print("-" * 80)

        # Header
        print(f"{'Period':<12} {'Trades':>8} {'Win Rate':>10} {'Net P&L':>14} {'Accretive':>12}")
        print("-" * 56)

        for stat in stats:
            accretive_str = "Yes" if stat.is_accretive else "No"
            pnl_str = f"${stat.net_pnl:+,.2f}"
            print(f"{stat.period:<12} {stat.trades:>8} {stat.win_rate:>9.1f}% {pnl_str:>14} {accretive_str:>12}")

        # Summary
        profitable_periods = sum(1 for s in stats if s.is_accretive)
        total_periods = len(stats)
        print("-" * 56)
        print(f"Profitable Periods: {profitable_periods}/{total_periods} ({profitable_periods / total_periods * 100:.1f}%)")

    def get_period_breakdown(self, period: str = "monthly") -> List[PeriodStats]:
        """
        Get statistics broken down by time period.

        Args:
            period: "monthly" or "weekly"

        Returns:
            List of PeriodStats for each period.
        """
        if not self.results.trades:
            return []

        # Group trades by period
        trades_by_period: Dict[str, List[CompletedTrade]] = defaultdict(list)

        for trade in self.results.trades:
            if period == "monthly":
                period_key = trade.exit_time.strftime("%Y-%m")
            elif period == "weekly":
                # ISO week format
                period_key = trade.exit_time.strftime("%Y-W%W")
            else:
                period_key = trade.exit_time.strftime("%Y-%m")

            trades_by_period[period_key].append(trade)

        # Calculate stats for each period
        stats = []
        for period_key in sorted(trades_by_period.keys()):
            period_trades = trades_by_period[period_key]

            wins = sum(1 for t in period_trades if t.is_winner)
            losses = len(period_trades) - wins
            gross_pnl = sum(t.gross_pnl for t in period_trades)
            total_fees = sum(t.entry_fees + t.exit_fee for t in period_trades)
            net_pnl = gross_pnl - total_fees

            stats.append(PeriodStats(
                period=period_key,
                trades=len(period_trades),
                wins=wins,
                losses=losses,
                win_rate=(wins / len(period_trades) * 100) if period_trades else 0,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                is_accretive=net_pnl > 0
            ))

        return stats

    def get_profitable_periods(self, period: str = "monthly") -> tuple[int, int, float]:
        """
        Count profitable (accretive) vs losing periods.

        Args:
            period: "monthly" or "weekly"

        Returns:
            Tuple of (profitable_count, total_count, percentage).
        """
        stats = self.get_period_breakdown(period)

        if not stats:
            return 0, 0, 0.0

        profitable = sum(1 for s in stats if s.is_accretive)
        total = len(stats)
        percentage = (profitable / total * 100) if total > 0 else 0

        return profitable, total, percentage

    def export_trades_csv(self, path: str) -> None:
        """
        Export trades to CSV file.

        Args:
            path: Output file path.
        """
        if not self.results.trades:
            return

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "trade_id", "pair", "side", "entry_price", "exit_price",
            "size", "entry_time", "exit_time", "gross_pnl", "net_pnl",
            "pnl_percent", "entry_fees", "exit_fee", "num_entries",
            "exit_reason", "is_winner"
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trade in self.results.trades:
                writer.writerow({
                    "trade_id": trade.trade_id,
                    "pair": trade.pair,
                    "side": trade.side,
                    "entry_price": f"{trade.entry_price:.2f}",
                    "exit_price": f"{trade.exit_price:.2f}",
                    "size": f"{trade.size:.8f}",
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "gross_pnl": f"{trade.gross_pnl:.2f}",
                    "net_pnl": f"{trade.net_pnl:.2f}",
                    "pnl_percent": f"{trade.pnl_percent:.2f}",
                    "entry_fees": f"{trade.entry_fees:.2f}",
                    "exit_fee": f"{trade.exit_fee:.2f}",
                    "num_entries": trade.num_entries,
                    "exit_reason": trade.exit_reason,
                    "is_winner": trade.is_winner
                })

        print(f"Exported {len(self.results.trades)} trades to {filepath}")

    def export_trades_json(self, path: str) -> None:
        """
        Export trades to JSON file.

        Args:
            path: Output file path.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.results.config.to_dict(),
            "summary": {
                "total_trades": self.results.metrics.total_trades,
                "win_rate": self.results.metrics.win_rate,
                "total_pnl": self.results.metrics.total_pnl,
                "total_fees": self.results.metrics.total_fees,
                "sharpe_ratio": self.results.metrics.sharpe_ratio,
                "sortino_ratio": self.results.metrics.sortino_ratio,
                "max_drawdown_percent": self.results.metrics.max_drawdown_percent,
                "profit_factor": self.results.metrics.profit_factor,
            },
            "trades": [trade.to_dict() for trade in self.results.trades]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Exported results to {filepath}")

    def export_equity_curve_csv(self, path: str) -> None:
        """
        Export equity curve to CSV file.

        Args:
            path: Output file path.
        """
        if not self.results.equity_curve:
            return

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "equity", "drawdown", "drawdown_percent"])

            for point in self.results.equity_curve:
                writer.writerow([
                    point.timestamp.isoformat(),
                    f"{point.equity:.2f}",
                    f"{point.drawdown:.2f}",
                    f"{point.drawdown_percent:.2f}"
                ])

        print(f"Exported equity curve to {filepath}")

    def get_summary_dict(self) -> dict:
        """Get summary as a dictionary."""
        config = self.results.config
        metrics = self.results.metrics

        final_capital = config.initial_capital + metrics.total_pnl - metrics.total_fees
        total_return = ((final_capital - config.initial_capital) / config.initial_capital) * 100

        profitable_periods, total_periods, period_pct = self.get_profitable_periods("monthly")

        return {
            "pair": config.pair,
            "period_start": config.start_date.isoformat(),
            "period_end": config.end_date.isoformat(),
            "period_days": config.period_days,
            "interval_minutes": config.interval,
            "initial_capital": config.initial_capital,
            "final_capital": final_capital,
            "total_return_percent": total_return,
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": metrics.win_rate,
            "gross_pnl": metrics.total_pnl,
            "total_fees": metrics.total_fees,
            "net_pnl": metrics.total_pnl - metrics.total_fees,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown": metrics.max_drawdown,
            "max_drawdown_percent": metrics.max_drawdown_percent,
            "profit_factor": metrics.profit_factor,
            "average_win": metrics.average_win,
            "average_loss": metrics.average_loss,
            "longest_win_streak": metrics.longest_win_streak,
            "longest_loss_streak": metrics.longest_loss_streak,
            "profitable_months": profitable_periods,
            "total_months": total_periods,
            "profitable_months_percent": period_pct,
        }
