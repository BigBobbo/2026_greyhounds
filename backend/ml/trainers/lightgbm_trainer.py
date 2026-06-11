"""LightGBM trainer implementation with isotonic calibration."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression as _PlattLR

from ml.monotonic_constraints import build_monotone_constraints
from ml.trainers.base import BaseTrainer, TrainResult


class LightGBMTrainer(BaseTrainer):
    def __init__(self, params: dict[str, Any], target_type: str = "classification"):
        super().__init__(params)
        self.target_type = target_type
        self.calibrator: _PlattLR | None = None
        self._target = params.get("_target", "win_prob")
        self._apply_monotone = params.get("apply_monotone_constraints", True)

        # Keep the raw params around; the model is instantiated in train()
        # once we know the feature-name order so monotone constraints can
        # be aligned column-by-column.
        model_params = {
            k: v for k, v in params.items()
            if k not in ("target_type", "_target", "apply_monotone_constraints")
        }
        model_params.setdefault("verbosity", -1)
        model_params.setdefault("random_state", 42)
        # No is_unbalance default — same probability-scale reasoning as
        # the XGBoost trainer; pass it explicitly if an experiment wants it.
        self._model_params = model_params
        self.model = None

    def train(self, X_train, y_train, X_val, y_val) -> TrainResult:
        model_params = dict(self._model_params)
        if self._apply_monotone:
            constraints = build_monotone_constraints(
                list(X_train.columns), self._target,
            )
            # LightGBM accepts a list of ints of same length as features.
            model_params["monotone_constraints"] = constraints

        if self.target_type == "classification":
            self.model = LGBMClassifier(**model_params)
        else:
            self.model = LGBMRegressor(**model_params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        from ml.evaluation import compute_metrics
        if self.target_type == "classification":
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_pred = self.model.predict(X_val)

            # Fit Platt scaling calibrator on validation set
            y_val_arr = np.asarray(y_val, dtype=float)
            if len(y_val_arr) >= 10 and len(np.unique(y_val_arr)) >= 2:
                self.calibrator = _PlattLR(C=1.0, max_iter=1000)
                log_odds = np.log(np.clip(val_proba, 1e-6, 1 - 1e-6) /
                                  (1 - np.clip(val_proba, 1e-6, 1 - 1e-6)))
                self.calibrator.fit(log_odds.reshape(-1, 1), y_val_arr)

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
