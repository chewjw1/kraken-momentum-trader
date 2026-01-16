"""
File-based persistence for state and trade history.
Suitable for Replit and local development.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..observability.logger import get_logger

logger = get_logger(__name__)


class FileStore:
    """
    File-based storage for application state.

    Features:
    - JSON-based storage
    - Atomic writes (write to temp, then rename)
    - Automatic backups
    - Directory creation
    """

    def __init__(self, base_path: str = "data"):
        """
        Initialize file store.

        Args:
            base_path: Base directory for data files.
        """
        self.base_path = Path(base_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create data directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, filename: str) -> Path:
        """Get full path for a file."""
        return self.base_path / filename

    def save(self, filename: str, data: dict) -> bool:
        """
        Save data to a JSON file.

        Args:
            filename: Name of the file.
            data: Data to save.

        Returns:
            True if successful.
        """
        filepath = self._get_path(filename)
        temp_path = filepath.with_suffix(".tmp")

        try:
            # Add metadata
            save_data = {
                "_meta": {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0"
                },
                **data
            }

            # Write to temp file first
            with open(temp_path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)

            # Atomic rename
            temp_path.replace(filepath)

            logger.debug(f"Saved data to {filename}", file=str(filepath))
            return True

        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            # Clean up temp file if exists
            if temp_path.exists():
                temp_path.unlink()
            return False

    def load(self, filename: str) -> Optional[dict]:
        """
        Load data from a JSON file.

        Args:
            filename: Name of the file.

        Returns:
            Loaded data or None if not found/error.
        """
        filepath = self._get_path(filename)

        if not filepath.exists():
            logger.debug(f"File not found: {filename}")
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Remove metadata before returning
            data.pop("_meta", None)

            logger.debug(f"Loaded data from {filename}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return None

    def exists(self, filename: str) -> bool:
        """Check if a file exists."""
        return self._get_path(filename).exists()

    def delete(self, filename: str) -> bool:
        """
        Delete a file.

        Args:
            filename: Name of the file.

        Returns:
            True if deleted (or didn't exist).
        """
        filepath = self._get_path(filename)

        try:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Deleted {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")
            return False

    def backup(self, filename: str) -> bool:
        """
        Create a backup of a file.

        Args:
            filename: Name of the file.

        Returns:
            True if backup created.
        """
        filepath = self._get_path(filename)

        if not filepath.exists():
            return False

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}.bak"
        backup_path = self.base_path / "backups" / backup_name

        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "r") as src:
                with open(backup_path, "w") as dst:
                    dst.write(src.read())
            logger.debug(f"Created backup: {backup_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False


class StateStore:
    """
    Specialized store for trading state management.
    """

    STATE_FILE = "state.json"
    TRADES_FILE = "trades.json"
    METRICS_FILE = "metrics.json"

    def __init__(self, base_path: str = "data"):
        """
        Initialize state store.

        Args:
            base_path: Base directory for data files.
        """
        self.store = FileStore(base_path)

    def save_state(self, state: dict) -> bool:
        """Save trading state."""
        return self.store.save(self.STATE_FILE, state)

    def load_state(self) -> Optional[dict]:
        """Load trading state."""
        return self.store.load(self.STATE_FILE)

    def save_trade(self, trade: dict) -> bool:
        """
        Append a trade to history.

        Args:
            trade: Trade data to save.
        """
        trades = self.load_trades() or []
        trades.append(trade)

        # Keep only last 1000 trades in file
        if len(trades) > 1000:
            trades = trades[-1000:]

        return self.store.save(self.TRADES_FILE, {"trades": trades})

    def load_trades(self) -> list:
        """Load trade history."""
        data = self.store.load(self.TRADES_FILE)
        return data.get("trades", []) if data else []

    def save_metrics(self, metrics: dict) -> bool:
        """Save performance metrics."""
        return self.store.save(self.METRICS_FILE, metrics)

    def load_metrics(self) -> Optional[dict]:
        """Load performance metrics."""
        return self.store.load(self.METRICS_FILE)

    def backup_all(self) -> bool:
        """Create backups of all files."""
        success = True
        for filename in [self.STATE_FILE, self.TRADES_FILE, self.METRICS_FILE]:
            if self.store.exists(filename):
                if not self.store.backup(filename):
                    success = False
        return success

    def clear_all(self) -> bool:
        """Clear all stored data."""
        success = True
        for filename in [self.STATE_FILE, self.TRADES_FILE, self.METRICS_FILE]:
            if not self.store.delete(filename):
                success = False
        return success


class TradeLogger:
    """
    Append-only trade log for audit purposes.
    """

    def __init__(self, base_path: str = "data"):
        """
        Initialize trade logger.

        Args:
            base_path: Base directory for log files.
        """
        self.base_path = Path(base_path) / "logs"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def log_trade(
        self,
        action: str,
        pair: str,
        side: str,
        price: float,
        amount: float,
        order_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a trade event.

        Args:
            action: Trade action (entry, exit, etc.).
            pair: Trading pair.
            side: Buy or sell.
            price: Trade price.
            amount: Trade amount.
            order_id: Optional order ID.
            **kwargs: Additional data.
        """
        timestamp = datetime.now(timezone.utc)
        log_file = self.base_path / f"trades_{timestamp.strftime('%Y%m')}.jsonl"

        entry = {
            "timestamp": timestamp.isoformat(),
            "action": action,
            "pair": pair,
            "side": side,
            "price": price,
            "amount": amount,
            "order_id": order_id,
            **kwargs
        }

        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def get_trades_for_date(self, date: datetime) -> list[dict]:
        """
        Get all trades for a specific date.

        Args:
            date: Date to filter by.

        Returns:
            List of trade entries.
        """
        log_file = self.base_path / f"trades_{date.strftime('%Y%m')}.jsonl"

        if not log_file.exists():
            return []

        trades = []
        date_str = date.strftime("%Y-%m-%d")

        try:
            with open(log_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("timestamp", "").startswith(date_str):
                        trades.append(entry)
        except Exception as e:
            logger.error(f"Failed to read trade log: {e}")

        return trades
