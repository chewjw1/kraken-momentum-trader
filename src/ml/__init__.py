"""
Machine Learning module for signal enhancement.

Provides ML-based confidence scoring for trade signals,
trained on historical data and continuously updated with new trades.
"""

from .features import FeatureExtractor, MLFeatures
from .predictor import SignalPredictor, MLPrediction
from .trainer import ModelTrainer
from .data_store import MLDataStore, TrainingSample
from .retrainer import AutoRetrainer

__all__ = [
    "FeatureExtractor",
    "MLFeatures",
    "SignalPredictor",
    "MLPrediction",
    "ModelTrainer",
    "MLDataStore",
    "TrainingSample",
    "AutoRetrainer",
]
