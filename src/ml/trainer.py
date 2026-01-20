"""
Model trainer for ML signal enhancement.

Trains models on historical/backtested data and supports retraining
with new trade outcomes.
"""

import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from .features import FeatureExtractor, MLFeatures
from .data_store import MLDataStore
from ..observability.logger import get_logger

logger = get_logger(__name__)

# Import sklearn - will be added to requirements
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. ML features disabled.")


class ModelTrainer:
    """
    Trains and manages ML models for signal prediction.

    Supports:
    - Initial training on backtested data
    - Retraining with new trade outcomes
    - Model persistence and versioning
    """

    def __init__(
        self,
        model_dir: str = "data/ml/models",
        data_store: Optional[MLDataStore] = None
    ):
        """
        Initialize trainer.

        Args:
            model_dir: Directory for saving models.
            data_store: Data store for training samples.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_store = data_store or MLDataStore()
        self.feature_extractor = FeatureExtractor()

        self.model = None
        self.scaler = None
        self.model_metadata = {}

    def train(
        self,
        min_samples: int = 100,
        test_size: float = 0.2,
        model_type: str = "gradient_boosting"
    ) -> dict:
        """
        Train model on stored training data.

        Args:
            min_samples: Minimum samples required.
            test_size: Fraction of data for testing.
            model_type: "gradient_boosting" or "random_forest".

        Returns:
            Training metrics dict.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for training. Run: pip install scikit-learn")

        # Get training data
        X, y = self.data_store.get_training_data(min_samples=min_samples)

        if len(X) == 0:
            raise ValueError(f"Insufficient training data. Need {min_samples} samples.")

        logger.info(f"Training on {len(X)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create model
        if model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )

        # Train
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "win_rate_actual": y_test.mean(),
            "win_rate_predicted": y_pred.mean(),
        }

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics["cv_score_mean"] = cv_scores.mean()
        metrics["cv_score_std"] = cv_scores.std()

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            feature_names = MLFeatures.feature_names()
            importances = dict(zip(feature_names, self.model.feature_importances_))
            metrics["feature_importance"] = dict(
                sorted(importances.items(), key=lambda x: x[1], reverse=True)
            )

        # Store metadata
        self.model_metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "total_samples": len(X),
            "metrics": metrics
        }

        logger.info(
            f"Training complete - Accuracy: {metrics['accuracy']:.2%}, "
            f"Precision: {metrics['precision']:.2%}, "
            f"F1: {metrics['f1']:.2%}"
        )

        return metrics

    def train_from_backtest(
        self,
        ohlcv_data: dict,
        strategy_params: dict,
        min_trades: int = 100
    ) -> dict:
        """
        Generate training data from backtesting and train model.

        Args:
            ohlcv_data: Dict of {pair: [(timestamp, o, h, l, c, v), ...]}.
            strategy_params: Strategy parameters for simulating trades.
            min_trades: Minimum trades to generate.

        Returns:
            Training metrics.
        """
        logger.info("Generating training data from backtest...")

        total_trades = 0

        for pair, candles in ohlcv_data.items():
            trades = self._simulate_trades(pair, candles, strategy_params)
            total_trades += len(trades)

            for trade in trades:
                self.data_store.add_historical_sample(**trade)

        self.data_store.save()
        logger.info(f"Generated {total_trades} simulated trades")

        if total_trades < min_trades:
            logger.warning(f"Only {total_trades} trades generated, need {min_trades}")

        return self.train(min_samples=min(total_trades, min_trades))

    def _simulate_trades(
        self,
        pair: str,
        candles: list,
        params: dict
    ) -> list[dict]:
        """
        Simulate trades on historical data to generate training samples.

        Args:
            pair: Trading pair.
            candles: List of (timestamp, open, high, low, close, volume).
            params: Strategy parameters.

        Returns:
            List of trade dicts with features and outcomes.
        """
        trades = []
        lookback = 50  # Need enough data for features

        rsi_oversold = params.get("rsi_oversold", 38)
        rsi_overbought = params.get("rsi_overbought", 62)
        trailing_stop_pct = params.get("trailing_stop_percent", 3.5)
        trailing_activation_pct = params.get("trailing_activation_percent", 2.5)

        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]
        timestamps = [c[0] for c in candles]

        in_position = False
        entry_price = 0
        entry_idx = 0
        entry_features = None
        peak_price = 0
        trailing_active = False

        for i in range(lookback, len(candles)):
            current_price = closes[i]
            current_time = timestamps[i]

            if isinstance(current_time, (int, float)):
                current_time = datetime.fromtimestamp(current_time, tz=timezone.utc)

            # Extract features at this point
            features = self.feature_extractor.extract_from_ohlcv(
                opens[:i+1],
                highs[:i+1],
                lows[:i+1],
                closes[:i+1],
                volumes[:i+1],
                current_time
            )

            if features is None:
                continue

            if not in_position:
                # Check for entry signal (RSI oversold)
                if features.rsi < rsi_oversold and features.volume_ratio > 1.0:
                    in_position = True
                    entry_price = current_price
                    entry_idx = i
                    entry_features = features
                    peak_price = current_price
                    trailing_active = False

            else:
                # Update peak price
                if current_price > peak_price:
                    peak_price = current_price

                # Check trailing stop activation
                gain_pct = ((current_price - entry_price) / entry_price) * 100
                if not trailing_active and gain_pct >= trailing_activation_pct:
                    trailing_active = True

                # Check exit conditions
                should_exit = False
                exit_reason = ""

                # Trailing stop hit
                if trailing_active:
                    drop_from_peak = ((peak_price - current_price) / peak_price) * 100
                    if drop_from_peak >= trailing_stop_pct:
                        should_exit = True
                        exit_reason = "trailing_stop"

                # RSI overbought
                if features.rsi > rsi_overbought:
                    should_exit = True
                    exit_reason = "rsi_overbought"

                # Max hold time (48 hours / 48 candles for hourly)
                if i - entry_idx > 48:
                    should_exit = True
                    exit_reason = "max_hold_time"

                if should_exit:
                    exit_price = current_price
                    pnl = exit_price - entry_price
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                    hold_duration = (i - entry_idx)  # hours for hourly candles

                    trade = {
                        "timestamp": timestamps[entry_idx] if isinstance(timestamps[entry_idx], datetime)
                                    else datetime.fromtimestamp(timestamps[entry_idx], tz=timezone.utc),
                        "pair": pair,
                        "signal_type": "buy",
                        "features": entry_features,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_percent": pnl_percent,
                        "hold_duration_hours": hold_duration
                    }
                    trades.append(trade)

                    in_position = False
                    entry_features = None

        return trades

    def save_model(self, filename: str = "signal_model.pkl") -> str:
        """
        Save trained model to file.

        Args:
            filename: Model filename.

        Returns:
            Path to saved model.
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        model_path = self.model_dir / filename
        scaler_path = self.model_dir / "scaler.pkl"
        metadata_path = self.model_dir / "metadata.json"

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save scaler
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        import json
        # Convert numpy types for JSON serialization
        metadata = self._serialize_metadata(self.model_metadata)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

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
        metadata_path = self.model_dir / "metadata.json"

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            if metadata_path.exists():
                import json
                with open(metadata_path, "r") as f:
                    self.model_metadata = json.load(f)

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _serialize_metadata(self, metadata: dict) -> dict:
        """Convert numpy types to Python types for JSON."""
        result = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                result[k] = self._serialize_metadata(v)
            elif isinstance(v, np.floating):
                result[k] = float(v)
            elif isinstance(v, np.integer):
                result[k] = int(v)
            elif isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        return result

    def get_model_info(self) -> dict:
        """Get information about current model."""
        if self.model is None:
            return {"status": "no_model"}

        return {
            "status": "loaded",
            "metadata": self.model_metadata,
            "data_stats": self.data_store.get_stats()
        }
