"""Scikit-learn trainer implementations (Logistic Regression, Random Forest) with isotonic calibration."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression as _PlattLR
from sklearn.preprocessing import StandardScaler

from ml.trainers.base import BaseTrainer, TrainResult


class SklearnTrainer(BaseTrainer):
    def __init__(self, params: dict[str, Any], algorithm: str = "logistic_regression", target_type: str = "classification"):
        super().__init__(params)
        self.algorithm = algorithm
        self.target_type = target_type
        self.scaler = StandardScaler()
        self.calibrator: _PlattLR | None = None

        # Strip pipeline-meta keys that create_trainer injects for the GBM
        # trainers (monotonic-constraint plumbing); sklearn estimators reject
        # unknown kwargs, so leaving them in broke every sklearn experiment.
        model_params = {
            k: v
            for k, v in params.items()
            if k not in ("algorithm", "target_type", "_target", "apply_monotone_constraints")
        }

        if algorithm == "logistic_regression":
            model_params.setdefault("max_iter", 1000)
            model_params.setdefault("random_state", 42)
            self.model = LogisticRegression(**model_params)
        elif algorithm == "random_forest":
            model_params.setdefault("random_state", 42)
            model_params.setdefault("n_jobs", 1)
            if target_type == "classification":
                self.model = RandomForestClassifier(**model_params)
            else:
                self.model = RandomForestRegressor(**model_params)
        else:
            raise ValueError(f"Unknown sklearn algorithm: {algorithm}")

    def train(self, X_train, y_train, X_val, y_val) -> TrainResult:
        # Scale features for logistic regression
        if self.algorithm == "logistic_regression":
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns,
            )
            X_val_scaled = pd.DataFrame(
                self.scaler.transform(X_val), index=X_val.index, columns=X_val.columns,
            )
        else:
            X_train_scaled, X_val_scaled = X_train, X_val

        self.model.fit(X_train_scaled, y_train)

        from ml.evaluation import compute_metrics
        if self.target_type == "classification":
            val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            val_pred = self.model.predict(X_val_scaled)

            # Fit Platt scaling calibrator on validation set
            y_val_arr = np.asarray(y_val, dtype=float)
            if len(y_val_arr) >= 10 and len(np.unique(y_val_arr)) >= 2:
                self.calibrator = _PlattLR(C=1.0, max_iter=1000)
                log_odds = np.log(np.clip(val_proba, 1e-6, 1 - 1e-6) /
                                  (1 - np.clip(val_proba, 1e-6, 1 - 1e-6)))
                self.calibrator.fit(log_odds.reshape(-1, 1), y_val_arr)

            metrics = compute_metrics(y_val, val_pred, val_proba, "classification")
        else:
            val_pred = self.model.predict(X_val_scaled)
            metrics = compute_metrics(y_val, val_pred, None, "regression")

        importance = self.get_feature_importance()
        return TrainResult(self.model, metrics, importance)

    def predict(self, X):
        if self.algorithm == "logistic_regression":
            X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        return self.model.predict(X)

    def predict_proba(self, X, calibrate: bool = True):
        if self.target_type != "classification":
            return None
        if self.algorithm == "logistic_regression":
            X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        raw = self.model.predict_proba(X)[:, 1]
        if calibrate and self.calibrator is not None:
            log_odds = np.log(np.clip(raw, 1e-6, 1 - 1e-6) /
                              (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
            return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]
        return raw

    def get_feature_importance(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            names = self.model.feature_names_in_ if hasattr(self.model, "feature_names_in_") else [f"f{i}" for i in range(len(self.model.feature_importances_))]
            return dict(zip(names, self.model.feature_importances_.tolist()))
        elif hasattr(self.model, "coef_"):
            names = self.model.feature_names_in_ if hasattr(self.model, "feature_names_in_") else [f"f{i}" for i in range(len(self.model.coef_[0]))]
            return dict(zip(names, np.abs(self.model.coef_[0]).tolist()))
        return {}
