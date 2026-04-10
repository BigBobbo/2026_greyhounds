"""XGBoost trainer implementation with isotonic calibration."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as _PlattLR
from xgboost import XGBClassifier, XGBRegressor

from ml.trainers.base import BaseTrainer, TrainResult


class XGBoostTrainer(BaseTrainer):
    def __init__(self, params: dict[str, Any], target_type: str = "classification"):
        super().__init__(params)
        self.target_type = target_type
        self.calibrator: _PlattLR | None = None

        model_params = {k: v for k, v in params.items() if k not in ("target_type",)}
        model_params.setdefault("verbosity", 0)
        model_params.setdefault("random_state", 42)

        if target_type == "classification":
            # Handle class imbalance: win is ~16.7% in 6-dog races
            # scale_pos_weight = neg_count / pos_count ≈ 5.0
            model_params.setdefault("scale_pos_weight", 5.0)
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

            # Fit Platt scaling calibrator on validation set
            self.calibrator = _PlattLR(C=1.0, max_iter=1000)
            log_odds = np.log(np.clip(val_proba, 1e-6, 1 - 1e-6) /
                              (1 - np.clip(val_proba, 1e-6, 1 - 1e-6)))
            self.calibrator.fit(log_odds.reshape(-1, 1), np.asarray(y_val, dtype=float))

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
                log_odds = np.log(np.clip(raw, 1e-6, 1 - 1e-6) /
                                  (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
                return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
            return raw
        return None

    def get_feature_importance(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_") and hasattr(self.model, "feature_names_in_"):
            return dict(zip(self.model.feature_names_in_, self.model.feature_importances_.tolist()))
        return {}
