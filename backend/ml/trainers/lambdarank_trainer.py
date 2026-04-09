"""LambdaRank trainer: Learning-to-Rank via LightGBM.

Instead of predicting each dog independently, this model learns to rank
all dogs within a race.  Training uses race-level groups so the model
sees the full competitive field at once.

At prediction time the raw ranking scores are converted to win
probabilities via softmax over the race.
"""

from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from ml.trainers.base import BaseTrainer, TrainResult


class LambdaRankTrainer(BaseTrainer):
    """LightGBM LambdaRank trainer for race-level ranking."""

    def __init__(self, params: dict[str, Any], target_type: str = "classification"):
        super().__init__(params)
        self.target_type = target_type  # always treated as ranking internally

        self.lgb_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3],
            "boosting_type": "gbdt",
            "num_leaves": params.get("num_leaves", 31),
            "learning_rate": params.get("learning_rate", 0.1),
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", -1),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "min_child_samples": params.get("min_child_samples", 20),
            "verbosity": -1,
            "random_state": params.get("random_state", 42),
            "importance_type": "gain",
        }
        self.model: lgb.LGBMRanker | None = None
        self._feature_names: list[str] = []

    def train(self, X_train, y_train, X_val, y_val,
              group_train=None, group_val=None) -> TrainResult:
        """Train the LambdaRank model.

        Args:
            group_train: array of group sizes (dogs per race) for training set.
                         If None, all entries are treated as one group (not ideal).
            group_val:   same for validation set.
        """
        self._feature_names = list(X_train.columns) if hasattr(X_train, "columns") else []

        # Build relevance labels from finish_position:
        # LambdaRank wants higher = more relevant.
        # We invert position: relevance = max_position - position
        # so 1st place gets highest relevance.
        y_train_rel = self._positions_to_relevance(y_train, group_train)
        y_val_rel = self._positions_to_relevance(y_val, group_val)

        n_estimators = self.lgb_params.pop("n_estimators", 200)

        self.model = lgb.LGBMRanker(
            n_estimators=n_estimators,
            **self.lgb_params,
        )
        self.model.fit(
            X_train, y_train_rel,
            group=group_train,
            eval_set=[(X_val, y_val_rel)],
            eval_group=[group_val],
            eval_at=[1, 3],
        )

        # Evaluate: compute ranking metrics on validation set
        val_scores = self.model.predict(X_val)
        metrics = self._compute_ranking_metrics(y_val, val_scores, group_val)

        importance = self.get_feature_importance()
        return TrainResult(self.model, metrics, importance)

    @staticmethod
    def _positions_to_relevance(positions, groups) -> np.ndarray:
        """Convert finish positions to relevance labels for LambdaRank.

        Within each race group, relevance = (max_runners - position + 1).
        This makes 1st place the most relevant, last place least.
        """
        positions = np.asarray(positions, dtype=float)

        if groups is None:
            max_pos = int(np.nanmax(positions))
            return np.clip(max_pos - positions + 1, 0, None).astype(int)

        relevance = np.zeros_like(positions, dtype=int)
        idx = 0
        for g_size in groups:
            group_pos = positions[idx:idx + g_size]
            max_pos = int(np.nanmax(group_pos)) if len(group_pos) > 0 else 6
            relevance[idx:idx + g_size] = np.clip(max_pos - group_pos + 1, 0, None).astype(int)
            idx += g_size

        return relevance

    @staticmethod
    def _compute_ranking_metrics(y_true, scores, groups) -> dict[str, float]:
        """Compute ranking-specific metrics."""
        y_true = np.asarray(y_true, dtype=float)
        metrics: dict[str, float] = {}

        # Top-1 accuracy: does the highest-scored dog actually win?
        correct_top1 = 0
        total_races = 0
        correct_top3 = 0

        if groups is not None:
            idx = 0
            for g_size in groups:
                if g_size == 0:
                    continue
                g_scores = scores[idx:idx + g_size]
                g_positions = y_true[idx:idx + g_size]

                pred_winner_idx = np.argmax(g_scores)
                actual_position = g_positions[pred_winner_idx]

                if actual_position == 1:
                    correct_top1 += 1
                if actual_position <= 3:
                    correct_top3 += 1
                total_races += 1
                idx += g_size

            if total_races > 0:
                metrics["top1_accuracy"] = correct_top1 / total_races
                metrics["top3_accuracy"] = correct_top3 / total_races
                metrics["total_races"] = float(total_races)

        return metrics

    def predict(self, X) -> np.ndarray:
        """Return raw ranking scores (higher = more likely to win)."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray | None:
        """Return raw scores — caller is responsible for softmax normalization
        within each race group."""
        if self.model is None:
            return None
        return self.model.predict(X)

    def scores_to_proba(self, scores: np.ndarray, group_sizes: list[int] | None = None) -> np.ndarray:
        """Convert raw ranking scores to win probabilities via softmax per race.

        Args:
            scores: Raw model scores for all entries.
            group_sizes: Number of dogs per race. If None, treat all as one race.

        Returns:
            Array of probabilities that sum to 1.0 within each race group.
        """
        proba = np.zeros_like(scores, dtype=float)

        if group_sizes is None:
            group_sizes = [len(scores)]

        idx = 0
        for g_size in group_sizes:
            g_scores = scores[idx:idx + g_size]
            # Softmax with numerical stability
            shifted = g_scores - np.max(g_scores)
            exp_scores = np.exp(shifted)
            proba[idx:idx + g_size] = exp_scores / exp_scores.sum()
            idx += g_size

        return proba

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            names = self._feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
            return dict(zip(names, self.model.feature_importances_.tolist()))
        return {}

    @staticmethod
    def get_default_params() -> dict[str, Any]:
        return {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
        }
