"""
Data store for ML training samples.

Stores feature vectors with outcomes for model training and retraining.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .features import MLFeatures
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
    Persistent storage for ML training data.

    Stores:
    - Completed trades with features and outcomes
    - Pending trades awaiting outcome
    """

    def __init__(self, base_path: str = "data/ml"):
        """
        Initialize data store.

        Args:
            base_path: Base directory for ML data files.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.samples_file = self.base_path / "training_samples.json"
        self.pending_file = self.base_path / "pending_trades.json"

        self._samples: list[TrainingSample] = []
        self._pending: dict[str, TrainingSample] = {}  # order_id -> sample

        self._load()

    def _load(self) -> None:
        """Load existing data from files."""
        # Load completed samples
        if self.samples_file.exists():
            try:
                with open(self.samples_file, "r") as f:
                    data = json.load(f)
                    self._samples = [TrainingSample(**s) for s in data.get("samples", [])]
                logger.info(f"Loaded {len(self._samples)} training samples")
            except Exception as e:
                logger.error(f"Failed to load training samples: {e}")
                self._samples = []

        # Load pending trades
        if self.pending_file.exists():
            try:
                with open(self.pending_file, "r") as f:
                    data = json.load(f)
                    self._pending = {
                        k: TrainingSample(**v)
                        for k, v in data.get("pending", {}).items()
                    }
                logger.info(f"Loaded {len(self._pending)} pending trades")
            except Exception as e:
                logger.error(f"Failed to load pending trades: {e}")
                self._pending = {}

    def _save_samples(self) -> None:
        """Save completed samples to file."""
        try:
            with open(self.samples_file, "w") as f:
                json.dump(
                    {"samples": [asdict(s) for s in self._samples]},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save training samples: {e}")

    def _save_pending(self) -> None:
        """Save pending trades to file."""
        try:
            with open(self.pending_file, "w") as f:
                json.dump(
                    {"pending": {k: asdict(v) for k, v in self._pending.items()}},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save pending trades: {e}")

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
        sample = TrainingSample(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pair=pair,
            signal_type=signal_type,
            features=asdict(features) if hasattr(features, '__dataclass_fields__') else features.__dict__,
            entry_price=entry_price
        )

        self._pending[order_id] = sample
        self._save_pending()

        logger.debug(f"Recorded entry for {pair} at {entry_price}")

    def record_exit(
        self,
        order_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ) -> Optional[TrainingSample]:
        """
        Record trade exit and compute outcome.

        Args:
            order_id: Order identifier from entry.
            exit_price: Exit price.
            pnl: Profit/loss in USD.
            pnl_percent: Profit/loss percentage.

        Returns:
            Completed TrainingSample or None if order not found.
        """
        if order_id not in self._pending:
            logger.warning(f"Order {order_id} not found in pending trades")
            return None

        sample = self._pending.pop(order_id)

        # Calculate hold duration
        entry_time = datetime.fromisoformat(sample.timestamp)
        exit_time = datetime.now(timezone.utc)
        hold_duration = (exit_time - entry_time).total_seconds() / 3600  # hours

        # Update sample with outcome
        sample.exit_price = exit_price
        sample.pnl = pnl
        sample.pnl_percent = pnl_percent
        sample.outcome = 1 if pnl > 0 else 0
        sample.hold_duration_hours = hold_duration

        # Add to completed samples
        self._samples.append(sample)

        # Save both files
        self._save_samples()
        self._save_pending()

        logger.info(
            f"Recorded exit for {sample.pair}: "
            f"{'WIN' if sample.outcome else 'LOSS'} {pnl_percent:.2f}%"
        )

        return sample

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
        sample = TrainingSample(
            timestamp=timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            pair=pair,
            signal_type=signal_type,
            features=asdict(features) if hasattr(features, '__dataclass_fields__') else features.__dict__,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            outcome=1 if pnl > 0 else 0,
            hold_duration_hours=hold_duration_hours
        )

        self._samples.append(sample)

    def save(self) -> None:
        """Force save all data."""
        self._save_samples()
        self._save_pending()

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
        samples = self._samples

        # Filter by pair if specified
        if pair:
            samples = [s for s in samples if s.pair == pair]

        # Filter only completed samples
        samples = [s for s in samples if s.outcome is not None]

        if len(samples) < min_samples:
            logger.warning(f"Insufficient samples: {len(samples)} < {min_samples}")
            return np.array([]), np.array([])

        # Extract features and labels
        feature_names = MLFeatures.feature_names()
        X = []
        y = []

        for sample in samples:
            features = sample.features
            feature_vector = [features.get(name, 0) for name in feature_names]
            X.append(feature_vector)
            y.append(sample.outcome)

        return np.array(X), np.array(y)

    def get_sample_count(self, pair: Optional[str] = None) -> int:
        """Get count of completed samples."""
        samples = self._samples
        if pair:
            samples = [s for s in samples if s.pair == pair]
        return len([s for s in samples if s.outcome is not None])

    def get_pending_count(self) -> int:
        """Get count of pending trades."""
        return len(self._pending)

    def get_stats(self) -> dict:
        """Get statistics about stored data."""
        completed = [s for s in self._samples if s.outcome is not None]
        wins = sum(1 for s in completed if s.outcome == 1)

        by_pair = {}
        for sample in completed:
            if sample.pair not in by_pair:
                by_pair[sample.pair] = {"total": 0, "wins": 0}
            by_pair[sample.pair]["total"] += 1
            if sample.outcome == 1:
                by_pair[sample.pair]["wins"] += 1

        return {
            "total_samples": len(completed),
            "pending_trades": len(self._pending),
            "overall_win_rate": (wins / len(completed) * 100) if completed else 0,
            "by_pair": by_pair
        }

    def clear_all(self) -> None:
        """Clear all stored data. Use with caution."""
        self._samples = []
        self._pending = {}
        self._save_samples()
        self._save_pending()
        logger.warning("All ML training data cleared")
