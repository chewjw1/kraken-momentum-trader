"""
Auto-retrainer for continuous model improvement.

Periodically retrains the ML model with new trade data.
"""

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

from .trainer import ModelTrainer
from .data_store import MLDataStore
from .predictor import SignalPredictor
from ..observability.logger import get_logger

logger = get_logger(__name__)


class AutoRetrainer:
    """
    Automatically retrains ML model based on triggers.

    Triggers:
    - After N new completed trades
    - On schedule (e.g., weekly)
    - When performance drops below threshold
    """

    def __init__(
        self,
        trainer: ModelTrainer,
        predictor: SignalPredictor,
        data_store: Optional[MLDataStore] = None,
        retrain_after_trades: int = 50,
        min_samples_for_retrain: int = 100,
        performance_threshold: float = 0.4,
        check_interval_seconds: int = 3600  # Check every hour
    ):
        """
        Initialize auto-retrainer.

        Args:
            trainer: Model trainer instance.
            predictor: Signal predictor to update after retraining.
            data_store: Data store for training samples.
            retrain_after_trades: Retrain after this many new trades.
            min_samples_for_retrain: Minimum samples required for retraining.
            performance_threshold: Retrain if recent win rate drops below this.
            check_interval_seconds: How often to check for retrain triggers.
        """
        self.trainer = trainer
        self.predictor = predictor
        self.data_store = data_store or MLDataStore()

        self.retrain_after_trades = retrain_after_trades
        self.min_samples_for_retrain = min_samples_for_retrain
        self.performance_threshold = performance_threshold
        self.check_interval_seconds = check_interval_seconds

        self._trades_since_retrain = 0
        self._last_retrain = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Callbacks
        self._on_retrain_complete: Optional[Callable[[dict], None]] = None

    def start(self) -> None:
        """Start the auto-retrainer background thread."""
        if self._running:
            logger.warning("Auto-retrainer already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"Auto-retrainer started (retrain every {self.retrain_after_trades} trades)"
        )

    def stop(self) -> None:
        """Stop the auto-retrainer."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("Auto-retrainer stopped")

    def record_trade_completed(self) -> None:
        """
        Notify retrainer that a trade completed.

        Should be called after each trade exit.
        """
        self._trades_since_retrain += 1
        logger.debug(
            f"Trade recorded. {self._trades_since_retrain}/{self.retrain_after_trades} until retrain"
        )

    def on_retrain_complete(self, callback: Callable[[dict], None]) -> None:
        """
        Set callback for when retraining completes.

        Args:
            callback: Function that receives training metrics dict.
        """
        self._on_retrain_complete = callback

    def _run_loop(self) -> None:
        """Main retrainer loop."""
        while self._running and not self._stop_event.is_set():
            try:
                should_retrain, reason = self._check_retrain_triggers()

                if should_retrain:
                    self._perform_retrain(reason)

                self._stop_event.wait(timeout=self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Auto-retrainer error: {e}")
                self._stop_event.wait(timeout=60)

    def _check_retrain_triggers(self) -> tuple[bool, str]:
        """
        Check if retraining should be triggered.

        Returns:
            Tuple of (should_retrain, reason).
        """
        # Check trade count trigger
        if self._trades_since_retrain >= self.retrain_after_trades:
            return True, f"Reached {self.retrain_after_trades} new trades"

        # Check if we have minimum samples
        sample_count = self.data_store.get_sample_count()
        if sample_count < self.min_samples_for_retrain:
            return False, f"Insufficient samples: {sample_count}"

        # Check performance trigger (recent win rate)
        stats = self.data_store.get_stats()
        recent_win_rate = self._calculate_recent_win_rate(20)  # Last 20 trades

        if recent_win_rate is not None and recent_win_rate < self.performance_threshold:
            return True, f"Performance dropped to {recent_win_rate:.1%}"

        return False, "No trigger"

    def _calculate_recent_win_rate(self, n_trades: int) -> Optional[float]:
        """Calculate win rate of last N trades."""
        # Get recent outcomes from SQLite
        recent_outcomes = self.data_store.db.get_recent_ml_outcomes(n_trades)

        if len(recent_outcomes) < n_trades:
            return None

        return sum(recent_outcomes) / len(recent_outcomes)

    def _perform_retrain(self, reason: str) -> None:
        """
        Perform model retraining.

        Args:
            reason: Reason for retraining.
        """
        logger.info(f"Starting model retrain: {reason}")

        try:
            # Train new model
            metrics = self.trainer.train(
                min_samples=self.min_samples_for_retrain
            )

            # Save model
            self.trainer.save_model()

            # Reload model in predictor
            self.predictor.load_model()

            # Reset counter
            self._trades_since_retrain = 0
            self._last_retrain = datetime.now(timezone.utc)

            logger.info(
                f"Retrain complete - Accuracy: {metrics['accuracy']:.1%}, "
                f"F1: {metrics['f1']:.1%}"
            )

            # Call callback if set
            if self._on_retrain_complete:
                self._on_retrain_complete(metrics)

        except Exception as e:
            logger.error(f"Retrain failed: {e}")

    def force_retrain(self) -> dict:
        """
        Force immediate retraining.

        Returns:
            Training metrics or error dict.
        """
        try:
            logger.info("Forcing model retrain")

            metrics = self.trainer.train(
                min_samples=self.min_samples_for_retrain
            )

            self.trainer.save_model()
            self.predictor.load_model()

            self._trades_since_retrain = 0
            self._last_retrain = datetime.now(timezone.utc)

            return metrics

        except Exception as e:
            logger.error(f"Forced retrain failed: {e}")
            return {"error": str(e)}

    def get_status(self) -> dict:
        """Get retrainer status."""
        return {
            "running": self._running,
            "trades_since_retrain": self._trades_since_retrain,
            "retrain_threshold": self.retrain_after_trades,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "total_samples": self.data_store.get_sample_count(),
            "min_samples_required": self.min_samples_for_retrain
        }
