"""LightGBM trainer implementation."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from ml.trainers.base import BaseTrainer, TrainResult


class LightGBMTrainer(BaseTrainer):
    def __init__(self, params: dict[str, Any], target_type: str = "classification"):
        super().__init__(params)
        self.target_type = target_type

        model_params = {k: v for k, v in params.items() if k not in ("target_type",)}
        model_params.setdefault("verbosity", -1)
        model_params.setdefault("random_state", 42)

        if target_type == "classification":
            self.model = LGBMClassifier(**model_params)
        else:
            self.model = LGBMRegressor(**model_params)

    def train(self, X_train, y_train, X_val, y_val) -> TrainResult:
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        from ml.evaluation import compute_metrics
        if self.target_type == "classification":
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_pred = self.model.predict(X_val)
            metrics = compute_metrics(y_val, val_pred, val_proba, "classification")
        else:
            val_pred = self.model.predict(X_val)
            metrics = compute_metrics(y_val, val_pred, None, "regression")

        importance = self.get_feature_importance()
        return TrainResult(self.model, metrics, importance)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.target_type == "classification":
            return self.model.predict_proba(X)[:, 1]
        return None

    def get_feature_importance(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_") and hasattr(self.model, "feature_names_in_"):
            return dict(zip(self.model.feature_names_in_, self.model.feature_importances_.tolist()))
        return {}
