"""
SQLite-based persistent storage for all trading data.

Provides ACID-compliant storage for:
- Trading state (positions, state machine)
- Historical trades
- Performance metrics
- ML training data
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

from ..observability.logger import get_logger

logger = get_logger(__name__)


class SQLiteStore:
    """
    Unified SQLite storage for all trading data.

    Single database file with tables for:
    - state: Current trading state and positions
    - trades: Historical trade log
    - metrics: Daily performance metrics
    - ml_samples: ML training samples
    - ml_pending: Pending trades awaiting exit
    """

    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"SQLite store initialized at {self.db_path}")

    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript("""
                -- Trading state (single row, updated in place)
                CREATE TABLE IF NOT EXISTS state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data JSON NOT NULL,
                    updated_at TEXT NOT NULL
                );

                -- Initialize state row if not exists
                INSERT OR IGNORE INTO state (id, data, updated_at)
                VALUES (1, '{}', datetime('now'));

                -- Historical trades
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    size_usd REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    pnl REAL,
                    pnl_percent REAL,
                    fees REAL DEFAULT 0,
                    num_entries INTEGER DEFAULT 1,
                    exit_reason TEXT,
                    metadata JSON,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair);
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
                CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

                -- Daily metrics
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    volume_usd REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                -- ML training samples (completed trades with features)
                CREATE TABLE IF NOT EXISTS ml_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    features JSON NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percent REAL NOT NULL,
                    outcome INTEGER NOT NULL,
                    hold_duration_hours REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_ml_samples_pair ON ml_samples(pair);
                CREATE INDEX IF NOT EXISTS idx_ml_samples_outcome ON ml_samples(outcome);

                -- ML pending trades (awaiting exit)
                CREATE TABLE IF NOT EXISTS ml_pending (
                    order_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    features JSON NOT NULL,
                    entry_price REAL NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Config/settings backup
                CREATE TABLE IF NOT EXISTS config_backup (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config JSON NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
            """)

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    # ========== State Management ==========

    def save_state(self, state: dict) -> None:
        """Save trading state."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE state SET data = ?, updated_at = ? WHERE id = 1",
                (json.dumps(state), datetime.now(timezone.utc).isoformat())
            )

    def load_state(self) -> Optional[dict]:
        """Load trading state."""
        with self._connection() as conn:
            row = conn.execute("SELECT data FROM state WHERE id = 1").fetchone()
            if row and row['data']:
                data = json.loads(row['data'])
                return data if data else None
            return None

    # ========== Trade Management ==========

    def save_trade(self, trade: dict) -> None:
        """Save a completed trade."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades
                (trade_id, pair, side, entry_price, exit_price, size, size_usd,
                 entry_time, exit_time, pnl, pnl_percent, fees, num_entries,
                 exit_reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get('trade_id'),
                trade.get('pair'),
                trade.get('side', 'long'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('size'),
                trade.get('size_usd'),
                trade.get('entry_time'),
                trade.get('exit_time'),
                trade.get('pnl'),
                trade.get('pnl_percent'),
                trade.get('fees', 0),
                trade.get('num_entries', 1),
                trade.get('reason'),
                json.dumps(trade.get('metadata', {}))
            ))

    def get_trades(
        self,
        pair: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> list[dict]:
        """Get historical trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if pair:
            query += " AND pair = ?"
            params.append(pair)
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_trade_count(self, pair: Optional[str] = None) -> int:
        """Get total number of trades."""
        with self._connection() as conn:
            if pair:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM trades WHERE pair = ?",
                    (pair,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as count FROM trades").fetchone()
            return row['count'] if row else 0

    def get_recent_trades(self, n: int = 20) -> list[dict]:
        """Get N most recent trades."""
        return self.get_trades(limit=n)

    # ========== Metrics Management ==========

    def save_daily_metrics(self, date: str, metrics: dict) -> None:
        """Save or update daily metrics."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_metrics
                (date, total_trades, wins, losses, pnl, fees, volume_usd, max_drawdown, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                metrics.get('total_trades', 0),
                metrics.get('wins', 0),
                metrics.get('losses', 0),
                metrics.get('pnl', 0),
                metrics.get('fees', 0),
                metrics.get('volume_usd', 0),
                metrics.get('max_drawdown', 0),
                datetime.now(timezone.utc).isoformat()
            ))

    def get_daily_metrics(self, date: str) -> Optional[dict]:
        """Get metrics for a specific date."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_metrics WHERE date = ?",
                (date,)
            ).fetchone()
            return dict(row) if row else None

    def get_metrics_range(self, start_date: str, end_date: str) -> list[dict]:
        """Get metrics for a date range."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_metrics WHERE date >= ? AND date <= ? ORDER BY date",
                (start_date, end_date)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_aggregate_metrics(self) -> dict:
        """Get aggregate metrics across all time."""
        with self._connection() as conn:
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
                    'total_pnl': 0, 'win_rate': 0
                }

            return {
                'total_trades': row['total_trades'],
                'wins': row['wins'] or 0,
                'losses': row['losses'] or 0,
                'total_pnl': row['total_pnl'] or 0,
                'total_fees': row['total_fees'] or 0,
                'avg_pnl_percent': row['avg_pnl_percent'] or 0,
                'best_trade': row['best_trade'] or 0,
                'worst_trade': row['worst_trade'] or 0,
                'win_rate': (row['wins'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0
            }

    # ========== ML Data Management ==========

    def add_ml_sample(
        self,
        timestamp: str,
        pair: str,
        signal_type: str,
        features: dict,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        hold_duration_hours: float
    ) -> None:
        """Add a completed ML training sample."""
        outcome = 1 if pnl > 0 else 0

        with self._connection() as conn:
            conn.execute("""
                INSERT INTO ml_samples
                (timestamp, pair, signal_type, features, entry_price, exit_price,
                 pnl, pnl_percent, outcome, hold_duration_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, pair, signal_type, json.dumps(features),
                entry_price, exit_price, pnl, pnl_percent, outcome, hold_duration_hours
            ))

    def record_ml_entry(
        self,
        order_id: str,
        pair: str,
        signal_type: str,
        features: dict,
        entry_price: float
    ) -> None:
        """Record a trade entry for later ML outcome tracking."""
        with self._connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ml_pending
                (order_id, timestamp, pair, signal_type, features, entry_price)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                order_id,
                datetime.now(timezone.utc).isoformat(),
                pair,
                signal_type,
                json.dumps(features),
                entry_price
            ))

    def record_ml_exit(
        self,
        order_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ) -> bool:
        """
        Record trade exit and move to completed samples.

        Returns:
            True if pending trade was found and processed.
        """
        with self._connection() as conn:
            # Get pending trade
            row = conn.execute(
                "SELECT * FROM ml_pending WHERE order_id = ?",
                (order_id,)
            ).fetchone()

            if not row:
                logger.warning(f"ML pending trade not found: {order_id}")
                return False

            # Calculate hold duration
            entry_time = datetime.fromisoformat(row['timestamp'])
            exit_time = datetime.now(timezone.utc)
            hold_duration = (exit_time - entry_time).total_seconds() / 3600

            # Add to samples
            self.add_ml_sample(
                timestamp=row['timestamp'],
                pair=row['pair'],
                signal_type=row['signal_type'],
                features=json.loads(row['features']),
                entry_price=row['entry_price'],
                exit_price=exit_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                hold_duration_hours=hold_duration
            )

            # Remove from pending
            conn.execute("DELETE FROM ml_pending WHERE order_id = ?", (order_id,))

            logger.info(f"ML sample recorded: {row['pair']} {'WIN' if pnl > 0 else 'LOSS'} {pnl_percent:.2f}%")
            return True

    def get_ml_training_data(
        self,
        min_samples: int = 0,
        pair: Optional[str] = None
    ) -> tuple[list[list[float]], list[int]]:
        """
        Get ML training data as feature vectors and labels.

        Returns:
            Tuple of (X features list, y labels list).
        """
        query = "SELECT features, outcome FROM ml_samples WHERE 1=1"
        params = []

        if pair:
            query += " AND pair = ?"
            params.append(pair)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

            if len(rows) < min_samples:
                logger.warning(f"Insufficient ML samples: {len(rows)} < {min_samples}")
                return [], []

            X = []
            y = []

            for row in rows:
                features = json.loads(row['features'])
                # Convert dict to ordered list based on feature names
                feature_vector = self._features_to_vector(features)
                X.append(feature_vector)
                y.append(row['outcome'])

            return X, y

    def _features_to_vector(self, features: dict) -> list[float]:
        """Convert feature dict to ordered vector."""
        # Must match MLFeatures.feature_names() order
        feature_names = [
            "rsi", "rsi_slope", "price_vs_ema_short", "price_vs_ema_long",
            "ema_crossover", "volume_ratio", "volume_trend", "price_change_1h",
            "price_change_4h", "price_change_24h", "volatility", "candle_body_ratio",
            "upper_wick_ratio", "lower_wick_ratio", "hour_of_day", "day_of_week",
            "trend_strength", "higher_highs", "lower_lows", "distance_to_high_20",
            "distance_to_low_20"
        ]
        return [features.get(name, 0) for name in feature_names]

    def get_ml_sample_count(self, pair: Optional[str] = None) -> int:
        """Get count of ML training samples."""
        with self._connection() as conn:
            if pair:
                row = conn.execute(
                    "SELECT COUNT(*) as count FROM ml_samples WHERE pair = ?",
                    (pair,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as count FROM ml_samples").fetchone()
            return row['count'] if row else 0

    def get_ml_pending_count(self) -> int:
        """Get count of pending ML trades."""
        with self._connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM ml_pending").fetchone()
            return row['count'] if row else 0

    def get_ml_stats(self) -> dict:
        """Get ML data statistics."""
        with self._connection() as conn:
            samples = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(outcome) as wins,
                    pair
                FROM ml_samples
                GROUP BY pair
            """).fetchall()

            pending = self.get_ml_pending_count()
            total = sum(row['total'] for row in samples)
            wins = sum(row['wins'] or 0 for row in samples)

            by_pair = {}
            for row in samples:
                by_pair[row['pair']] = {
                    'total': row['total'],
                    'wins': row['wins'] or 0
                }

            return {
                'total_samples': total,
                'pending_trades': pending,
                'overall_win_rate': (wins / total * 100) if total > 0 else 0,
                'by_pair': by_pair
            }

    def get_recent_ml_outcomes(self, n: int = 20) -> list[int]:
        """Get outcomes of last N ML samples."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT outcome FROM ml_samples ORDER BY id DESC LIMIT ?",
                (n,)
            ).fetchall()
            return [row['outcome'] for row in rows]

    def clear_ml_data(self) -> None:
        """Clear all ML data. Use with caution."""
        with self._connection() as conn:
            conn.execute("DELETE FROM ml_samples")
            conn.execute("DELETE FROM ml_pending")
        logger.warning("All ML data cleared")

    # ========== Backup & Migration ==========

    def backup_config(self, config: dict) -> None:
        """Save a config backup."""
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO config_backup (config) VALUES (?)",
                (json.dumps(config),)
            )

    def export_to_json(self, output_dir: str = "data/backup") -> dict:
        """Export all data to JSON files for backup."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exports = {}

        with self._connection() as conn:
            # Export trades
            trades = conn.execute("SELECT * FROM trades").fetchall()
            trades_data = [dict(row) for row in trades]
            with open(output_path / "trades.json", "w") as f:
                json.dump(trades_data, f, indent=2)
            exports['trades'] = len(trades_data)

            # Export ML samples
            samples = conn.execute("SELECT * FROM ml_samples").fetchall()
            samples_data = [dict(row) for row in samples]
            with open(output_path / "ml_samples.json", "w") as f:
                json.dump(samples_data, f, indent=2)
            exports['ml_samples'] = len(samples_data)

            # Export daily metrics
            metrics = conn.execute("SELECT * FROM daily_metrics").fetchall()
            metrics_data = [dict(row) for row in metrics]
            with open(output_path / "daily_metrics.json", "w") as f:
                json.dump(metrics_data, f, indent=2)
            exports['daily_metrics'] = len(metrics_data)

        logger.info(f"Data exported to {output_path}: {exports}")
        return exports

    def backup_all(self) -> bool:
        """Create a backup of the database."""
        try:
            backup_path = self.db_path.parent / "backups"
            backup_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = backup_path / f"trading_{timestamp}.db"

            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_file)

            logger.info(f"Database backed up to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    def import_from_json(self, json_dir: str = "data") -> dict:
        """
        Import data from existing JSON files (migration from old format).

        Args:
            json_dir: Directory containing JSON files.

        Returns:
            Dict with import counts.
        """
        json_path = Path(json_dir)
        imports = {}

        # Import trades from trades.json
        trades_file = json_path / "trades.json"
        if trades_file.exists():
            with open(trades_file, "r") as f:
                data = json.load(f)
                trades = data.get('trades', []) if isinstance(data, dict) else data
                for trade in trades:
                    self.save_trade(trade)
                imports['trades'] = len(trades)

        # Import ML samples from ml/training_samples.json
        ml_file = json_path / "ml" / "training_samples.json"
        if ml_file.exists():
            with open(ml_file, "r") as f:
                data = json.load(f)
                samples = data.get('samples', [])
                with self._connection() as conn:
                    for sample in samples:
                        if sample.get('outcome') is not None:
                            conn.execute("""
                                INSERT INTO ml_samples
                                (timestamp, pair, signal_type, features, entry_price,
                                 exit_price, pnl, pnl_percent, outcome, hold_duration_hours)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                sample.get('timestamp'),
                                sample.get('pair'),
                                sample.get('signal_type'),
                                json.dumps(sample.get('features', {})),
                                sample.get('entry_price'),
                                sample.get('exit_price'),
                                sample.get('pnl'),
                                sample.get('pnl_percent'),
                                sample.get('outcome'),
                                sample.get('hold_duration_hours')
                            ))
                imports['ml_samples'] = len(samples)

        # Import pending trades
        pending_file = json_path / "ml" / "pending_trades.json"
        if pending_file.exists():
            with open(pending_file, "r") as f:
                data = json.load(f)
                pending = data.get('pending', {})
                for order_id, trade in pending.items():
                    self.record_ml_entry(
                        order_id=order_id,
                        pair=trade.get('pair'),
                        signal_type=trade.get('signal_type'),
                        features=trade.get('features', {}),
                        entry_price=trade.get('entry_price')
                    )
                imports['ml_pending'] = len(pending)

        logger.info(f"Data imported from {json_path}: {imports}")
        return imports
