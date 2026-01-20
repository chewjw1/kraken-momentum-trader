"""
ML predictor for live signal enhancement.

Loads trained model and provides confidence scores for trade signals.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .features import FeatureExtractor, MLFeatures
from .data_store import MLDataStore
from ..strategy.base_strategy import MarketData
from ..observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLPrediction:
    """ML model prediction result."""
    signal: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    win_probability: float  # Probability of profitable trade
    feature_contributions: Optional[dict] = None  # Top contributing features


class SignalPredictor:
    """
    Predicts trade signal confidence using trained ML model.

    Integrates with the trading strategy to enhance or filter signals.
    """

    def __init__(
        self,
        model_dir: str = "data/ml/models",
        confidence_threshold: float = 0.6,
        enabled: bool = True,
        db_path: str = "data/trading.db"
    ):
        """
        Initialize predictor.

        Args:
            model_dir: Directory containing saved model.
            confidence_threshold: Minimum confidence to approve signal.
            enabled: Whether ML predictions are enabled.
            db_path: Path to SQLite database.
        """
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled

        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self.data_store = MLDataStore(db_path=db_path)

        self._model_loaded = False

    def load_model(self, filename: str = "signal_model.pkl") -> bool:
        """
        Load model from file.

        Args:
            filename: Model filename.

        Returns:
            True if loaded successfully.
        """
        model_path = self.model_dir / filename
        scaler_path = self.model_dir / "scaler.pkl"

        if not model_path.exists():
            logger.info(f"No ML model found at {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            self._model_loaded = True
            logger.info("ML model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False

    def predict(self, market_data: MarketData) -> Optional[MLPrediction]:
        """
        Predict signal confidence for current market conditions.

        Args:
            market_data: Current market data.

        Returns:
            MLPrediction or None if prediction fails.
        """
        if not self.enabled:
            return None

        if not self._model_loaded:
            if not self.load_model():
                return None

        try:
            # Extract features
            features = self.feature_extractor.extract(market_data)
            if features is None:
                logger.warning("Feature extraction failed")
                return None

            # Prepare input
            X = features.to_array().reshape(1, -1)

            # Scale if scaler available
            if self.scaler is not None:
                X = self.scaler.transform(X)

            # Get prediction
            prob = self.model.predict_proba(X)[0]
            win_prob = prob[1]  # Probability of class 1 (profitable)

            # Determine signal based on probability
            if win_prob >= self.confidence_threshold:
                signal = "buy"
                confidence = win_prob
            elif win_prob <= (1 - self.confidence_threshold):
                signal = "sell"
                confidence = 1 - win_prob
            else:
                signal = "hold"
                confidence = 0.5

            # Get feature contributions if available
            contributions = None
            if hasattr(self.model, "feature_importances_"):
                feature_names = MLFeatures.feature_names()
                feature_values = features.to_array()
                importances = self.model.feature_importances_

                # Top 5 contributing features
                top_indices = np.argsort(importances)[-5:][::-1]
                contributions = {
                    feature_names[i]: {
                        "importance": float(importances[i]),
                        "value": float(feature_values[i])
                    }
                    for i in top_indices
                }

            return MLPrediction(
                signal=signal,
                confidence=confidence,
                win_probability=win_prob,
                feature_contributions=contributions
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None

    def should_take_signal(
        self,
        signal_type: str,
        market_data: MarketData
    ) -> tuple[bool, float, str]:
        """
        Determine if a trading signal should be acted upon.

        Args:
            signal_type: "buy" or "sell" from rule-based strategy.
            market_data: Current market data.

        Returns:
            Tuple of (should_trade, confidence, reason).
        """
        if not self.enabled or not self._model_loaded:
            # ML disabled - pass through all signals
            return True, 1.0, "ML disabled"

        prediction = self.predict(market_data)

        if prediction is None:
            # Prediction failed - pass through signal with warning
            return True, 0.5, "ML prediction failed"

        # For buy signals, check if ML agrees
        if signal_type == "buy":
            if prediction.win_probability >= self.confidence_threshold:
                return True, prediction.win_probability, f"ML confidence: {prediction.win_probability:.1%}"
            else:
                return False, prediction.win_probability, f"ML confidence too low: {prediction.win_probability:.1%}"

        # For sell signals, always allow (exiting positions is generally safer)
        if signal_type == "sell":
            return True, prediction.confidence, "Exit signal approved"

        return True, 0.5, "Unknown signal type"

    def record_trade_entry(
        self,
        order_id: str,
        pair: str,
        signal_type: str,
        market_data: MarketData,
        entry_price: float
    ) -> None:
        """
        Record trade entry for later outcome tracking.

        Args:
            order_id: Unique order identifier.
            pair: Trading pair.
            signal_type: "buy" or "sell".
            market_data: Market data at entry.
            entry_price: Entry price.
        """
        features = self.feature_extractor.extract(market_data)
        if features:
            self.data_store.record_entry(
                order_id=order_id,
                pair=pair,
                signal_type=signal_type,
                features=features,
                entry_price=entry_price
            )

    def record_trade_exit(
        self,
        order_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ) -> None:
        """
        Record trade exit for training data.

        Args:
            order_id: Order identifier from entry.
            exit_price: Exit price.
            pnl: Profit/loss.
            pnl_percent: Profit/loss percentage.
        """
        self.data_store.record_exit(
            order_id=order_id,
            exit_price=exit_price,
            pnl=pnl,
            pnl_percent=pnl_percent
        )

    def get_stats(self) -> dict:
        """Get predictor statistics."""
        return {
            "enabled": self.enabled,
            "model_loaded": self._model_loaded,
            "confidence_threshold": self.confidence_threshold,
            "data_stats": self.data_store.get_stats()
        }

    @property
    def is_ready(self) -> bool:
        """Check if predictor is ready for predictions."""
        return self.enabled and self._model_loaded
