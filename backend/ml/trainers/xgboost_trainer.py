"""XGBoost trainer implementation with isotonic calibration."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier, XGBRegressor

from ml.trainers.base import BaseTrainer, TrainResult


class XGBoostTrainer(BaseTrainer):
    def __init__(self, params: dict[str, Any], target_type: str = "classification"):
        super().__init__(params)
        self.target_type = target_type
        self.calibrator: IsotonicRegression | None = None

        model_params = {k: v for k, v in params.items() if k not in ("target_type",)}
        model_params.setdefault("verbosity", 0)
        model_params.setdefault("random_state", 42)

        if target_type == "classification":
            self.model = XGBClassifier(**model_params)
        else:
            self.model = XGBRegressor(**model_params)

    def train(self, X_train, y_train, X_val, y_val) -> TrainResult:
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        from ml.evaluation import compute_metrics
        if self.target_type == "classification":
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_pred = self.model.predict(X_val)

            # Fit isotonic calibrator on validation set
            self.calibrator = IsotonicRegression(
                y_min=0.01, y_max=0.99, out_of_bounds="clip",
            )
            self.calibrator.fit(val_proba, np.asarray(y_val, dtype=float))

            metrics = compute_metrics(y_val, val_pred, val_proba, "classification")
        else:
            val_pred = self.model.predict(X_val)
            metrics = compute_metrics(y_val, val_pred, None, "regression")

        importance = self.get_feature_importance()
        return TrainResult(self.model, metrics, importance)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X, calibrate: bool = True):
        if self.target_type == "classification":
            raw = self.model.predict_proba(X)[:, 1]
            if calibrate and self.calibrator is not None:
                return self.calibrator.predict(raw)
            return raw
        return None

    def get_feature_importance(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_") and hasattr(self.model, "feature_names_in_"):
            return dict(zip(self.model.feature_names_in_, self.model.feature_importances_.tolist()))
        return {}
