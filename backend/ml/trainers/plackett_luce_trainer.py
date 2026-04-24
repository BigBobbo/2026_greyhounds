"""Plackett-Luce trainer with Henery place/show correction.

Trains a per-dog scoring function f(features) -> score with a true
Plackett-Luce listwise loss applied to the top-3 finishing order:

    L = -log P(1st) - log P(2nd | 1st) - log P(3rd | 1st, 2nd)

Each factor is a softmax over the field that has not yet been "consumed"
by an earlier finisher.  This is implemented as a custom LightGBM
objective (gradient + diagonal Hessian) so we can reuse the existing
gradient-boosted-trees infrastructure.

After training, the validation set is used to fit Henery (lambda_2,
lambda_3) score-discount parameters by grid search; these are stored on
the trainer and applied at inference time when computing place / show /
exacta / trifecta probabilities.

Why a separate trainer (vs. LambdaRank):
    * LambdaRank optimises NDCG, an ordinal proxy.  Plackett-Luce
      optimises the actual ordering likelihood, giving better-calibrated
      win probabilities out of the box.
    * Henery correction directly addresses the well-documented Harville
      bias against longshots in place / show markets.
    * The same scoring function naturally produces P(1st), P(2nd) and
      P(3rd) via closed-form marginalisation — no extra heads needed.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.position_distribution import (
    HeneryLambdas,
    fit_henery_lambdas,
    split_by_groups,
)
from ml.trainers.base import BaseTrainer, TrainResult


class PlackettLuceTrainer(BaseTrainer):
    """LightGBM booster trained with a Plackett-Luce top-3 listwise loss."""

    def __init__(self, params: dict[str, Any], target_type: str = "classification"):
        super().__init__(params)
        # target_type is accepted for API parity with sibling trainers; PL is
        # always a ranking objective regardless of the experiment's nominal target.
        self.target_type = target_type

        self.lgb_params: dict[str, Any] = {
            "boosting_type": "gbdt",
            "num_leaves": params.get("num_leaves", 31),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_depth": params.get("max_depth", -1),
            "feature_fraction": params.get("colsample_bytree", 0.8),
            "bagging_fraction": params.get("subsample", 0.8),
            "bagging_freq": 1,
            "min_child_samples": params.get("min_child_samples", 20),
            "verbosity": -1,
            "seed": params.get("random_state", 42),
        }
        self.n_estimators: int = int(params.get("n_estimators", 300))

        self.model: lgb.Booster | None = None
        self._feature_names: list[str] = []
        self.henery: HeneryLambdas = HeneryLambdas(1.0, 1.0)

    # ---------------------------------------------------------------- training

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        group_train: list[int] | np.ndarray | None = None,
        group_val: list[int] | np.ndarray | None = None,
    ) -> TrainResult:
        if group_train is None or group_val is None:
            raise ValueError(
                "PlackettLuceTrainer requires group_train and group_val "
                "(race-grouping is intrinsic to the loss)."
            )

        self._feature_names = list(X_train.columns)

        y_train_arr = np.asarray(y_train, dtype=float)
        y_val_arr = np.asarray(y_val, dtype=float)
        group_train = list(group_train)
        group_val = list(group_val)

        # The label LightGBM stores is the finish position; the custom
        # objective consumes it directly via train_set.get_label().
        train_set = lgb.Dataset(
            X_train, label=y_train_arr, group=group_train, free_raw_data=False
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val_arr,
            group=group_val,
            reference=train_set,
            free_raw_data=False,
        )

        # The custom objective closes over `group_train` for fast slicing.
        # LightGBM rebuilds the gradient on every iteration, so we can't rely
        # on dataset.get_group() returning anything at predict-time — we capture
        # the grouping by row range.
        train_groups = np.asarray(group_train, dtype=int)
        val_groups = np.asarray(group_val, dtype=int)

        def _objective(preds: np.ndarray, dataset: lgb.Dataset):
            labels = dataset.get_label()
            # LightGBM passes preds for whichever dataset is being scored.
            if dataset is val_set:
                groups = val_groups
            else:
                groups = train_groups
            return _plackett_luce_grad_hess(preds, labels, groups)

        def _eval_metric(preds: np.ndarray, dataset: lgb.Dataset):
            labels = dataset.get_label()
            if dataset is val_set:
                groups = val_groups
            else:
                groups = train_groups
            nll = _plackett_luce_nll(preds, labels, groups)
            return ("pl_nll", float(nll), False)  # lower is better

        params = dict(self.lgb_params)
        params["objective"] = _objective

        self.model = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=self.n_estimators,
            valid_sets=[val_set],
            valid_names=["val"],
            feval=_eval_metric,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        # Fit Henery (lambda_2, lambda_3) on the validation set so the
        # place/show distribution at inference time matches observed
        # frequencies rather than the Harville-biased default.
        val_scores = self.model.predict(X_val)
        score_groups = split_by_groups(val_scores, group_val)
        position_groups = split_by_groups(y_val_arr, group_val)
        self.henery = fit_henery_lambdas(score_groups, position_groups)

        metrics = self._compute_metrics(val_scores, y_val_arr, group_val)
        importance = self.get_feature_importance()
        return TrainResult(self.model, metrics, importance)

    # --------------------------------------------------------------- inference

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return raw scores.  The caller should apply softmax per race
        (use ``scores_to_proba``) or call ``position_distributions`` for
        the full per-position breakdown."""
        if self.model is None:
            return None
        return self.model.predict(X)

    def scores_to_proba(
        self,
        scores: np.ndarray,
        group_sizes: list[int] | None = None,
        calibrate: bool = True,
    ) -> np.ndarray:
        """Convert raw scores to per-race win probabilities via softmax.

        The ``calibrate`` argument is accepted for API parity with the
        LambdaRank trainer; for PL the softmax is the model's intrinsic
        win probability so no extra calibration layer is applied.
        """
        from ml.position_distribution import _stable_softmax

        scores = np.asarray(scores, dtype=float)
        if group_sizes is None:
            group_sizes = [scores.size]
        out = np.zeros_like(scores, dtype=float)
        idx = 0
        for g in group_sizes:
            if g <= 0:
                continue
            out[idx : idx + g] = _stable_softmax(scores[idx : idx + g])
            idx += g
        return out

    def position_distributions(
        self,
        scores: np.ndarray,
        group_sizes: list[int] | None = None,
    ) -> list[np.ndarray]:
        """Return per-race (n_dogs x 4) position-probability matrices.

        Column 0 = P(1st), col 1 = P(2nd), col 2 = P(3rd), col 3 = P(4+).
        Henery lambdas learned at training time are applied automatically.
        """
        from ml.position_distribution import position_probabilities

        scores = np.asarray(scores, dtype=float)
        if group_sizes is None:
            group_sizes = [scores.size]

        out: list[np.ndarray] = []
        idx = 0
        for g in group_sizes:
            if g <= 0:
                out.append(np.zeros((0, 4)))
                continue
            out.append(
                position_probabilities(scores[idx : idx + g], lambdas=self.henery)
            )
            idx += g
        return out

    # ----------------------------------------------------------- bookkeeping

    def get_feature_importance(self) -> dict[str, float]:
        if self.model is None:
            return {}
        importances = self.model.feature_importance(importance_type="gain")
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names, importances.tolist()))

    @staticmethod
    def _compute_metrics(
        scores: np.ndarray,
        positions: np.ndarray,
        groups: list[int],
    ) -> dict[str, float]:
        """Top-1 accuracy, top-3 accuracy, and validation PL log-likelihood."""
        idx = 0
        correct_top1 = 0
        correct_top3 = 0
        total_races = 0
        for g in groups:
            if g <= 0:
                continue
            g_scores = scores[idx : idx + g]
            g_pos = positions[idx : idx + g]
            picked = int(np.argmax(g_scores))
            if g_pos[picked] == 1:
                correct_top1 += 1
            if g_pos[picked] <= 3:
                correct_top3 += 1
            total_races += 1
            idx += g

        nll = _plackett_luce_nll(scores, positions, np.asarray(groups, dtype=int))

        if total_races == 0:
            return {"pl_nll": float(nll)}
        return {
            "top1_accuracy": correct_top1 / total_races,
            "top3_accuracy": correct_top3 / total_races,
            "total_races": float(total_races),
            "pl_nll": float(nll),
        }

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


# ---------------------------------------------------------------------------
# Plackett-Luce loss: gradient + diagonal Hessian
# ---------------------------------------------------------------------------


def _plackett_luce_grad_hess(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and diagonal Hessian for the top-k Plackett-Luce loss.

    Per race with scores s and finish positions p, with (k1, k2, k3)
    being the dogs that finished 1st/2nd/3rd:

        L = sum_{t=1..k} [-s_{k_t} + log sum_{i in S_t} exp(s_i)]

        grad_i = sum_{t : i in S_t} [softmax(s; S_t)_i - 1{i = k_t}]
        hess_i = sum_{t : i in S_t} [softmax(s; S_t)_i * (1 - softmax(s; S_t)_i)]

    where S_t is the set of dogs not yet "consumed" by an earlier finisher.

    Smaller finish positions are better; ties (e.g. dead heats coded with
    duplicate positions) are broken by the dog's score-array order.
    """
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)

    idx = 0
    for g_size in groups:
        g_size = int(g_size)
        if g_size <= 0:
            continue
        s = preds[idx : idx + g_size]
        pos = labels[idx : idx + g_size]

        # Determine the finishing order (top-k indices into the local race).
        k = min(top_k, g_size)
        # argsort is ascending — smallest position (=1st place) first.
        # Dogs without a recorded finish position will have NaN; sort them last.
        order = np.argsort(np.where(np.isnan(pos), np.inf, pos))[:k]

        remaining = np.ones(g_size, dtype=bool)
        for finisher_idx in order:
            finisher_idx = int(finisher_idx)
            if not remaining[finisher_idx]:
                continue
            sub_scores = s[remaining]
            shifted = sub_scores - sub_scores.max()
            ex = np.exp(np.clip(shifted, -100.0, 0.0))
            total = ex.sum()
            if total <= 0:
                p = np.full_like(sub_scores, 1.0 / sub_scores.size)
            else:
                p = ex / total

            # Distribute gradients back into the full race-sized vector.
            g_local = np.zeros(g_size)
            g_local[remaining] = p
            g_local[finisher_idx] -= 1.0
            grad[idx : idx + g_size] += g_local

            h_local = np.zeros(g_size)
            h_local[remaining] = p * (1.0 - p)
            hess[idx : idx + g_size] += h_local

            remaining[finisher_idx] = False

        idx += g_size

    # LightGBM expects strictly positive Hessian for stability; floor it.
    hess = np.maximum(hess, 1e-6)
    return grad, hess


def _plackett_luce_nll(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    top_k: int = 3,
) -> float:
    """Negative log-likelihood of the observed top-k orderings."""
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=float)
    nll = 0.0
    eps = 1e-12

    idx = 0
    for g_size in groups:
        g_size = int(g_size)
        if g_size <= 0:
            continue
        s = preds[idx : idx + g_size]
        pos = labels[idx : idx + g_size]
        k = min(top_k, g_size)
        order = np.argsort(np.where(np.isnan(pos), np.inf, pos))[:k]

        remaining = np.ones(g_size, dtype=bool)
        for finisher_idx in order:
            finisher_idx = int(finisher_idx)
            if not remaining[finisher_idx]:
                continue
            sub_scores = s[remaining]
            shifted = sub_scores - sub_scores.max()
            ex = np.exp(np.clip(shifted, -100.0, 0.0))
            total = ex.sum()
            if total <= 0:
                p_finisher = 1.0 / sub_scores.size
            else:
                # Index of finisher within remaining sub-array
                local_indices = np.where(remaining)[0]
                local_idx = int(np.where(local_indices == finisher_idx)[0][0])
                p_finisher = ex[local_idx] / total
            nll -= float(np.log(max(p_finisher, eps)))
            remaining[finisher_idx] = False

        idx += g_size

    return nll
