"""Abstract base trainer interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class TrainResult:
    """Result of a training run."""

    def __init__(
        self,
        model: Any,
        metrics: dict[str, float],
        feature_importance: dict[str, float] | None = None,
    ):
        self.model = model
        self.metrics = metrics
        self.feature_importance = feature_importance


class BaseTrainer(ABC):
    """Abstract trainer interface for all ML algorithms."""

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.model = None

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> TrainResult:
        """Train the model and return results."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Generate probability predictions (classification only). Returns None for regression."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        ...

    @staticmethod
    def get_default_params(algorithm: str) -> dict[str, Any]:
        """Get default hyperparameters for an algorithm."""
        defaults = {
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
            },
            "lightgbm": {
                "n_estimators": 200,
                "max_depth": -1,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
            },
            "lambdarank": {
                "n_estimators": 300,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
            },
            "plackett_luce": {
                "n_estimators": 300,
                "num_leaves": 31,
                "learning_rate": 0.05,
                "max_depth": -1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
            },
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 1000,
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
        }
        return defaults.get(algorithm, {})
