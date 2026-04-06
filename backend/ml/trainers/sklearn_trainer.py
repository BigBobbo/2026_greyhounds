"""Scikit-learn trainer implementations (Logistic Regression, Random Forest)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ml.trainers.base import BaseTrainer, TrainResult


class SklearnTrainer(BaseTrainer):
    def __init__(self, params: dict[str, Any], algorithm: str = "logistic_regression", target_type: str = "classification"):
        super().__init__(params)
        self.algorithm = algorithm
        self.target_type = target_type
        self.scaler = StandardScaler()

        model_params = {k: v for k, v in params.items() if k not in ("algorithm", "target_type")}

        if algorithm == "logistic_regression":
            model_params.setdefault("max_iter", 1000)
            model_params.setdefault("random_state", 42)
            self.model = LogisticRegression(**model_params)
        elif algorithm == "random_forest":
            model_params.setdefault("random_state", 42)
            model_params.setdefault("n_jobs", -1)
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

    def predict_proba(self, X):
        if self.target_type != "classification":
            return None
        if self.algorithm == "logistic_regression":
            X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            names = self.model.feature_names_in_ if hasattr(self.model, "feature_names_in_") else [f"f{i}" for i in range(len(self.model.feature_importances_))]
            return dict(zip(names, self.model.feature_importances_.tolist()))
        elif hasattr(self.model, "coef_"):
            names = self.model.feature_names_in_ if hasattr(self.model, "feature_names_in_") else [f"f{i}" for i in range(len(self.model.coef_[0]))]
            return dict(zip(names, np.abs(self.model.coef_[0]).tolist()))
        return {}
