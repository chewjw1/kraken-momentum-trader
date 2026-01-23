"""
Read-only data service for the dashboard.
Uses separate connections to avoid conflicts with the running trader.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional


class DashboardDataService:
    """Read-only data service for dashboard."""

    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)

    def _get_connection(self):
        """Get a new read-only connection."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=5
        )
        conn.row_factory = sqlite3.Row
        return conn

    def get_current_state(self) -> dict:
        """Get current trading state including positions."""
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT data FROM state WHERE id = 1").fetchone()
                if row and row['data']:
                    return json.loads(row['data'])
        except Exception:
            pass
        return {}

    def get_aggregate_metrics(self) -> dict:
        """Get aggregate metrics across all trades."""
        try:
            with self._get_connection() as conn:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                        SUM(pnl) as total_pnl,
                        SUM(fees) as total_fees,
                        AVG(pnl_percent) as avg_pnl_percent,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM trades WHERE exit_price IS NOT NULL
                """).fetchone()

                if not row or row['total_trades'] == 0:
                    return {
                        'total_trades': 0, 'wins': 0, 'losses': 0,
                        'total_pnl': 0, 'total_fees': 0, 'win_rate': 0,
                        'profit_factor': 0, 'avg_pnl_percent': 0,
                        'best_trade': 0, 'worst_trade': 0
                    }

                win_rate = (row['wins'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0

                winning = conn.execute(
                    "SELECT SUM(pnl) as gross_profit FROM trades WHERE pnl > 0 AND exit_price IS NOT NULL"
                ).fetchone()
                losing = conn.execute(
                    "SELECT SUM(ABS(pnl)) as gross_loss FROM trades WHERE pnl <= 0 AND exit_price IS NOT NULL"
                ).fetchone()

                gross_profit = winning['gross_profit'] or 0
                gross_loss = losing['gross_loss'] or 0
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

                return {
                    'total_trades': row['total_trades'],
                    'wins': row['wins'] or 0,
                    'losses': row['losses'] or 0,
                    'total_pnl': row['total_pnl'] or 0,
                    'total_fees': row['total_fees'] or 0,
                    'avg_pnl_percent': row['avg_pnl_percent'] or 0,
                    'best_trade': row['best_trade'] or 0,
                    'worst_trade': row['worst_trade'] or 0,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor
                }
        except Exception:
            return {
                'total_trades': 0, 'wins': 0, 'losses': 0,
                'total_pnl': 0, 'total_fees': 0, 'win_rate': 0,
                'profit_factor': 0, 'avg_pnl_percent': 0,
                'best_trade': 0, 'worst_trade': 0
            }

    def get_trades(
        self,
        pair: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """Get trades with pagination and filtering."""
        try:
            query = "SELECT * FROM trades WHERE exit_price IS NOT NULL"
            count_query = "SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL"
            params = []
            count_params = []

            if pair:
                query += " AND pair = ?"
                count_query += " AND pair = ?"
                params.append(pair)
                count_params.append(pair)
            if start_date:
                query += " AND entry_time >= ?"
                count_query += " AND entry_time >= ?"
                params.append(start_date)
                count_params.append(start_date)
            if end_date:
                query += " AND entry_time <= ?"
                count_query += " AND entry_time <= ?"
                params.append(end_date + "T23:59:59")
                count_params.append(end_date + "T23:59:59")

            query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            with self._get_connection() as conn:
                total = conn.execute(count_query, count_params).fetchone()[0]
                rows = conn.execute(query, params).fetchall()
                return [dict(row) for row in rows], total
        except Exception:
            return [], 0

    def get_recent_trades(self, n: int = 10) -> list[dict]:
        """Get N most recent trades."""
        trades, _ = self.get_trades(limit=n)
        return trades

    def get_daily_metrics(self, days: int = 30) -> list[dict]:
        """Get daily metrics for the last N days."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM daily_metrics
                    ORDER BY date DESC LIMIT ?
                """, (days,)).fetchall()
                return [dict(row) for row in reversed(rows)]
        except Exception:
            return []

    def get_equity_curve(self, initial_capital: float = 10000) -> list[dict]:
        """Calculate equity curve from trades."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT exit_time, pnl, fees
                    FROM trades
                    WHERE exit_price IS NOT NULL
                    ORDER BY exit_time ASC
                """).fetchall()

                equity = initial_capital
                curve = [{'date': 'Start', 'equity': initial_capital}]

                for row in rows:
                    equity += (row['pnl'] or 0) - (row['fees'] or 0)
                    date_str = row['exit_time'][:10] if row['exit_time'] else 'Unknown'
                    curve.append({
                        'date': date_str,
                        'equity': round(equity, 2)
                    })

                return curve
        except Exception:
            return [{'date': 'Start', 'equity': initial_capital}]

    def get_ml_stats(self) -> dict:
        """Get ML training statistics."""
        try:
            with self._get_connection() as conn:
                row = conn.execute("""
                    SELECT COUNT(*) as total, SUM(outcome) as wins
                    FROM ml_samples
                """).fetchone()

                pending = conn.execute("SELECT COUNT(*) FROM ml_pending").fetchone()[0]

                return {
                    'total_samples': row['total'] or 0,
                    'win_rate': (row['wins'] / row['total'] * 100) if row['total'] and row['total'] > 0 else 0,
                    'pending_trades': pending
                }
        except Exception:
            return {'total_samples': 0, 'win_rate': 0, 'pending_trades': 0}

    def get_pairs_summary(self) -> list[dict]:
        """Get performance summary by trading pair."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT
                        pair,
                        COUNT(*) as trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(pnl) as total_pnl,
                        AVG(pnl_percent) as avg_pnl_percent
                    FROM trades
                    WHERE exit_price IS NOT NULL
                    GROUP BY pair
                    ORDER BY total_pnl DESC
                """).fetchall()
                return [dict(row) for row in rows]
        except Exception:
            return []
