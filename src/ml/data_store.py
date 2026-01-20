"""
Data store for ML training samples.

Stores feature vectors with outcomes for model training and retraining.
Uses SQLite for persistent storage.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .features import MLFeatures
from ..persistence.sqlite_store import SQLiteStore
from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingSample:
    """A single training sample with features and outcome."""
    timestamp: str  # ISO format
    pair: str
    signal_type: str  # "buy" or "sell"
    features: dict  # Feature values
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    outcome: Optional[int] = None  # 1 = profitable, 0 = loss, None = pending
    hold_duration_hours: Optional[float] = None


class MLDataStore:
    """
    Persistent storage for ML training data using SQLite.

    Stores:
    - Completed trades with features and outcomes
    - Pending trades awaiting outcome
    """

    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize data store with SQLite backend.

        Args:
            db_path: Path to SQLite database.
        """
        self.db = SQLiteStore(db_path)
        logger.info(f"ML data store initialized with SQLite: {db_path}")

    def record_entry(
        self,
        order_id: str,
        pair: str,
        signal_type: str,
        features: MLFeatures,
        entry_price: float
    ) -> None:
        """
        Record a trade entry for later outcome tracking.

        Args:
            order_id: Unique order identifier.
            pair: Trading pair.
            signal_type: "buy" or "sell".
            features: ML features at entry time.
            entry_price: Entry price.
        """
        features_dict = asdict(features) if hasattr(features, '__dataclass_fields__') else features.__dict__

        self.db.record_ml_entry(
            order_id=order_id,
            pair=pair,
            signal_type=signal_type,
            features=features_dict,
            entry_price=entry_price
        )

        logger.debug(f"Recorded entry for {pair} at {entry_price}")

    def record_exit(
        self,
        order_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ) -> bool:
        """
        Record trade exit and compute outcome.

        Args:
            order_id: Order identifier from entry.
            exit_price: Exit price.
            pnl: Profit/loss in USD.
            pnl_percent: Profit/loss percentage.

        Returns:
            True if recorded successfully.
        """
        return self.db.record_ml_exit(
            order_id=order_id,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent
        )

    def add_historical_sample(
        self,
        timestamp: datetime,
        pair: str,
        signal_type: str,
        features: MLFeatures,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        hold_duration_hours: float
    ) -> None:
        """
        Add a historical/backtested sample directly.

        Args:
            timestamp: Time of entry.
            pair: Trading pair.
            signal_type: "buy" or "sell".
            features: ML features at entry time.
            entry_price: Entry price.
            exit_price: Exit price.
            pnl: Profit/loss.
            pnl_percent: Profit/loss percentage.
            hold_duration_hours: How long position was held.
        """
        features_dict = asdict(features) if hasattr(features, '__dataclass_fields__') else features.__dict__
        timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp

        self.db.add_ml_sample(
            timestamp=timestamp_str,
            pair=pair,
            signal_type=signal_type,
            features=features_dict,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            hold_duration_hours=hold_duration_hours
        )

    def save(self) -> None:
        """Force save all data (no-op for SQLite, auto-committed)."""
        pass  # SQLite auto-commits

    def get_training_data(
        self,
        min_samples: int = 0,
        pair: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get training data as numpy arrays.

        Args:
            min_samples: Minimum samples required (returns empty if not met).
            pair: Optional filter by pair.

        Returns:
            Tuple of (features_array, labels_array).
        """
        X, y = self.db.get_ml_training_data(min_samples=min_samples, pair=pair)

        if not X:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

    def get_sample_count(self, pair: Optional[str] = None) -> int:
        """Get count of completed samples."""
        return self.db.get_ml_sample_count(pair=pair)

    def get_pending_count(self) -> int:
        """Get count of pending trades."""
        return self.db.get_ml_pending_count()

    def get_stats(self) -> dict:
        """Get statistics about stored data."""
        return self.db.get_ml_stats()

    def clear_all(self) -> None:
        """Clear all stored data. Use with caution."""
        self.db.clear_ml_data()
        logger.warning("All ML training data cleared")
