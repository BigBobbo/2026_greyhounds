"""
Autoresearch: autonomous ML experiment loop for greyhound prediction.

Inspired by Andrej Karpathy's autoresearch. Proposes experiment configurations
(hyperparameters, feature subsets, algorithms), trains, evaluates, and keeps
only changes that beat the current best result. Logs everything to the
existing Experiment table for full reproducibility.

Usage:
    from ml.autoresearch import AutoResearchLoop
    loop = AutoResearchLoop(db, feature_ids=[1, 2, 3], objective="betting_kelly_roi")
    loop.run(max_experiments=100)
"""

import copy
import logging
import random
import time
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.models.experiment import Experiment
from ml.dataset_builder import build_dataset
from ml.evaluation import compute_betting_metrics, compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mutation strategies — each returns a new (algorithm, hyperparams, feature_ids)
# ---------------------------------------------------------------------------

ALGORITHMS = ["xgboost", "lightgbm", "lambdarank", "random_forest"]

# Hyperparameter search spaces per algorithm
SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "xgboost": {
        "n_estimators": {"type": "int", "low": 50, "high": 600},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.3},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 30},
        "scale_pos_weight": {"type": "float", "low": 1.0, "high": 10.0},
        "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        "reg_alpha": {"type": "float_log", "low": 1e-5, "high": 10.0},
        "reg_lambda": {"type": "float_log", "low": 1e-5, "high": 10.0},
    },
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 50, "high": 600},
        "num_leaves": {"type": "int", "low": 15, "high": 127},
        "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.3},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 5, "high": 60},
        "reg_alpha": {"type": "float_log", "low": 1e-5, "high": 10.0},
        "reg_lambda": {"type": "float_log", "low": 1e-5, "high": 10.0},
        "max_depth": {"type": "int", "low": -1, "high": 15},
    },
    "lambdarank": {
        "n_estimators": {"type": "int", "low": 100, "high": 800},
        "num_leaves": {"type": "int", "low": 15, "high": 127},
        "learning_rate": {"type": "float_log", "low": 0.005, "high": 0.2},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.4, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 5, "high": 60},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 50, "high": 600},
        "max_depth": {"type": "int", "low": 3, "high": 25},
        "min_samples_split": {"type": "int", "low": 2, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 15},
    },
}


def _sample_param(spec: dict[str, Any]) -> float | int:
    """Sample a single hyperparameter from its spec."""
    if spec["type"] == "int":
        return random.randint(spec["low"], spec["high"])
    elif spec["type"] == "float":
        return round(random.uniform(spec["low"], spec["high"]), 6)
    elif spec["type"] == "float_log":
        log_low = np.log(spec["low"])
        log_high = np.log(spec["high"])
        return round(float(np.exp(random.uniform(log_low, log_high))), 6)
    return spec.get("low", 0)


def _random_hyperparams(algorithm: str) -> dict[str, Any]:
    """Generate a fully random hyperparameter config for the given algorithm."""
    space = SEARCH_SPACES.get(algorithm, {})
    return {k: _sample_param(v) for k, v in space.items()}


def _mutate_hyperparams(
    params: dict[str, Any],
    algorithm: str,
    n_mutations: int = 2,
) -> dict[str, Any]:
    """Mutate 1-3 hyperparameters from the current best, keeping the rest."""
    space = SEARCH_SPACES.get(algorithm, {})
    if not space:
        return params

    new_params = copy.deepcopy(params)
    keys = list(space.keys())
    to_mutate = random.sample(keys, min(n_mutations, len(keys)))

    for k in to_mutate:
        spec = space[k]
        if spec["type"] == "int":
            current = new_params.get(k, (spec["low"] + spec["high"]) // 2)
            # Perturb by ~20% of range
            delta = max(1, int(0.2 * (spec["high"] - spec["low"])))
            new_val = current + random.randint(-delta, delta)
            new_params[k] = max(spec["low"], min(spec["high"], new_val))
        elif spec["type"] in ("float", "float_log"):
            new_params[k] = _sample_param(spec)

    return new_params


def _mutate_features(
    current_ids: list[int],
    all_ids: list[int],
    n_changes: int = 1,
) -> list[int]:
    """Add or remove features from the current set."""
    result = list(current_ids)
    available_to_add = [f for f in all_ids if f not in result]

    for _ in range(n_changes):
        action = random.choice(["add", "remove", "swap"])
        if action == "add" and available_to_add:
            pick = random.choice(available_to_add)
            result.append(pick)
            available_to_add.remove(pick)
        elif action == "remove" and len(result) > 3:
            pick = random.choice(result)
            result.remove(pick)
            available_to_add.append(pick)
        elif action == "swap" and available_to_add and len(result) > 3:
            remove = random.choice(result)
            add = random.choice(available_to_add)
            result.remove(remove)
            result.append(add)
            available_to_add.remove(add)
            available_to_add.append(remove)

    return sorted(set(result))


# ---------------------------------------------------------------------------
# Proposal strategies
# ---------------------------------------------------------------------------

def _propose_hyperparam_mutation(state: dict) -> dict:
    """Mutate hyperparams of the current best config."""
    new_params = _mutate_hyperparams(
        state["best_hyperparams"],
        state["best_algorithm"],
        n_mutations=random.randint(1, 3),
    )
    return {
        "algorithm": state["best_algorithm"],
        "hyperparameters": new_params,
        "feature_set": list(state["best_features"]),
        "strategy": "hyperparam_mutation",
    }


def _propose_random_hyperparams(state: dict) -> dict:
    """Fully random hyperparams on the current best algorithm."""
    return {
        "algorithm": state["best_algorithm"],
        "hyperparameters": _random_hyperparams(state["best_algorithm"]),
        "feature_set": list(state["best_features"]),
        "strategy": "random_hyperparams",
    }


def _propose_algorithm_switch(state: dict) -> dict:
    """Try a different algorithm with its default-ish params."""
    other = [a for a in ALGORITHMS if a != state["best_algorithm"]]
    algo = random.choice(other)
    return {
        "algorithm": algo,
        "hyperparameters": _random_hyperparams(algo),
        "feature_set": list(state["best_features"]),
        "strategy": f"algorithm_switch_{algo}",
    }


def _propose_feature_mutation(state: dict) -> dict:
    """Add/remove/swap features from the current best set."""
    new_features = _mutate_features(
        state["best_features"],
        state["all_feature_ids"],
        n_changes=random.randint(1, 2),
    )
    return {
        "algorithm": state["best_algorithm"],
        "hyperparameters": dict(state["best_hyperparams"]),
        "feature_set": new_features,
        "strategy": "feature_mutation",
    }


def _propose_combined_mutation(state: dict) -> dict:
    """Mutate both hyperparams and features simultaneously."""
    new_params = _mutate_hyperparams(
        state["best_hyperparams"],
        state["best_algorithm"],
        n_mutations=random.randint(1, 2),
    )
    new_features = _mutate_features(
        state["best_features"],
        state["all_feature_ids"],
        n_changes=1,
    )
    return {
        "algorithm": state["best_algorithm"],
        "hyperparameters": new_params,
        "feature_set": new_features,
        "strategy": "combined_mutation",
    }


# Weighted strategy selection — exploitation-heavy with exploration bursts
STRATEGIES = [
    (_propose_hyperparam_mutation, 0.40),
    (_propose_random_hyperparams, 0.10),
    (_propose_algorithm_switch, 0.10),
    (_propose_feature_mutation, 0.25),
    (_propose_combined_mutation, 0.15),
]


def _select_strategy():
    """Weighted random strategy selection."""
    funcs, weights = zip(*STRATEGIES)
    return random.choices(funcs, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

# Objectives and their direction (True = higher is better)
OBJECTIVE_DIRECTIONS: dict[str, bool] = {
    "betting_kelly_roi": True,
    "betting_kelly_pnl": True,
    "betting_top_pick_roi": True,
    "betting_top_pick_pnl": True,
    "betting_top_pick_strike_rate": True,
    "betting_value_roi": True,
    "test_log_loss": False,
    "test_brier_score": False,
    "test_roc_auc": True,
    "test_accuracy": True,
    "test_top1_accuracy": True,
}


def _is_improvement(
    new_value: float,
    best_value: float,
    higher_is_better: bool,
) -> bool:
    """Check if new_value is strictly better than best_value."""
    if higher_is_better:
        return new_value > best_value
    return new_value < best_value


def _run_single_experiment(
    db: Session,
    proposal: dict,
    target: str,
    split_config: dict | None,
) -> tuple[dict[str, float] | None, float | None]:
    """
    Run a single training experiment without saving to DB.

    Returns (all_metrics, training_duration) or (None, None) on failure.
    """
    from app.services.training_service import create_trainer

    algorithm = proposal["algorithm"]
    hyperparams = proposal["hyperparameters"]
    feature_ids = proposal["feature_set"]
    is_ranking = algorithm == "lambdarank"

    try:
        build_target = "finish_position" if is_ranking else target
        dataset = build_dataset(
            db,
            feature_ids=feature_ids,
            target=build_target,
            split_config=split_config,
            only_complete=False,
        )

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]
        group_train = dataset.get("group_train")
        group_val = dataset.get("group_val")
        group_test = dataset.get("group_test")
        meta_test = dataset.get("meta_test")

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning("Empty dataset, skipping")
            return None, None

        trainer = create_trainer(algorithm, hyperparams, target)

        start = time.time()
        if is_ranking:
            trainer.train(X_train, y_train, X_val, y_val,
                          group_train=group_train, group_val=group_val)
        else:
            trainer.train(X_train, y_train, X_val, y_val)
        duration = time.time() - start

        # Evaluate on test set
        target_type = "regression" if target == "finish_time" else "classification"
        test_pred = trainer.predict(X_test)

        if is_ranking:
            test_proba = trainer.scores_to_proba(test_pred, group_test)
            test_metrics = trainer._compute_ranking_metrics(y_test, test_pred, group_test)
            y_test_binary = (y_test == 1).astype(float)
            test_pred_binary = np.zeros_like(y_test_binary)
            idx = 0
            for g_size in (group_test or [len(test_pred)]):
                g_scores = test_pred[idx:idx + g_size]
                winner_idx = np.argmax(g_scores)
                test_pred_binary[idx + winner_idx] = 1
                idx += g_size
            cls_metrics = compute_metrics(
                y_test_binary, test_pred_binary, test_proba, "classification",
            )
            test_metrics.update(cls_metrics)
        else:
            test_proba = trainer.predict_proba(X_test)
            test_metrics = compute_metrics(y_test, test_pred, test_proba, target_type)

        all_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

        # Betting metrics
        can_eval_betting = (
            (target_type == "classification" or is_ranking)
            and test_proba is not None
            and meta_test is not None
        )
        if can_eval_betting:
            try:
                y_binary = (
                    (y_test == 1).astype(float).values
                    if is_ranking
                    else y_test.values
                )
                betting = compute_betting_metrics(
                    y_binary,
                    test_proba,
                    meta_test["sp_decimal"].values,
                    meta_test["race_id"].values,
                )
                all_metrics["betting_top_pick_pnl"] = betting["top_pick_pnl"]
                all_metrics["betting_top_pick_roi"] = betting["top_pick_roi"]
                all_metrics["betting_top_pick_strike_rate"] = betting["top_pick_strike_rate"]
                all_metrics["betting_value_pnl"] = betting["value_bet_pnl"]
                all_metrics["betting_value_roi"] = betting["value_bet_roi"]
                all_metrics["betting_kelly_pnl"] = betting.get("kelly_pnl", 0)
                all_metrics["betting_kelly_roi"] = betting.get("kelly_roi", 0)
            except Exception as e:
                logger.warning("Betting metrics failed: %s", e)

        return all_metrics, duration

    except Exception as e:
        logger.error("Experiment failed: %s", e, exc_info=True)
        return None, None


class AutoResearchLoop:
    """
    Autonomous ML experiment loop.

    Proposes experiment configurations, trains, evaluates, and keeps only
    improvements. All experiments are logged to the Experiment table.

    Args:
        db: SQLAlchemy session
        feature_ids: List of feature_definition IDs to use as the starting set
        objective: Metric to optimize (key from OBJECTIVE_DIRECTIONS)
        algorithm: Starting algorithm (default: "lightgbm")
        target: Prediction target (default: "win_prob")
        split_config: Train/val/test split config
        all_feature_ids: Full list of available feature IDs for feature search.
            If None, uses feature_ids as both starting set and full set.
    """

    def __init__(
        self,
        db: Session,
        feature_ids: list[int],
        objective: str = "betting_kelly_roi",
        algorithm: str = "lightgbm",
        target: str = "win_prob",
        split_config: dict | None = None,
        all_feature_ids: list[int] | None = None,
    ):
        if objective not in OBJECTIVE_DIRECTIONS:
            raise ValueError(
                f"Unknown objective '{objective}'. "
                f"Choose from: {list(OBJECTIVE_DIRECTIONS.keys())}"
            )

        self.db = db
        self.objective = objective
        self.higher_is_better = OBJECTIVE_DIRECTIONS[objective]
        self.target = target
        self.split_config = split_config or {}
        self.all_feature_ids = all_feature_ids or list(feature_ids)

        # Initial state from defaults
        from ml.trainers.base import BaseTrainer
        self.state: dict[str, Any] = {
            "best_algorithm": algorithm,
            "best_hyperparams": BaseTrainer.get_default_params(algorithm),
            "best_features": list(feature_ids),
            "best_score": float("-inf") if self.higher_is_better else float("inf"),
            "all_feature_ids": self.all_feature_ids,
        }

        # Run log
        self.history: list[dict[str, Any]] = []

    def run(
        self,
        max_experiments: int = 100,
        patience: int = 20,
        on_improvement: Any = None,
    ) -> dict[str, Any]:
        """
        Run the autoresearch loop.

        Args:
            max_experiments: Maximum number of experiments to run.
            patience: Stop after this many experiments without improvement.
            on_improvement: Optional callback(experiment_number, score, proposal)
                called each time a new best is found.

        Returns:
            Summary dict with best config, score, and full history.
        """
        logger.info(
            "Starting autoresearch: objective=%s (%s), max=%d, patience=%d",
            self.objective,
            "maximize" if self.higher_is_better else "minimize",
            max_experiments,
            patience,
        )

        # Run baseline experiment first
        baseline = {
            "algorithm": self.state["best_algorithm"],
            "hyperparameters": dict(self.state["best_hyperparams"]),
            "feature_set": list(self.state["best_features"]),
            "strategy": "baseline",
        }
        logger.info("[0/%d] Running baseline...", max_experiments)
        metrics, duration = _run_single_experiment(
            self.db, baseline, self.target, self.split_config,
        )
        if metrics and self.objective in metrics:
            self.state["best_score"] = metrics[self.objective]
            self._log_experiment(0, baseline, metrics, duration, accepted=True)
            self._save_experiment(baseline, metrics, duration, is_best=True, trial_num=0)
            logger.info(
                "[0/%d] Baseline %s = %.4f",
                max_experiments, self.objective, self.state["best_score"],
            )
        else:
            logger.warning("Baseline failed or missing objective metric")
            self._log_experiment(0, baseline, metrics, duration, accepted=False)

        no_improvement_count = 0

        for i in range(1, max_experiments + 1):
            if no_improvement_count >= patience:
                logger.info(
                    "Stopping: no improvement for %d experiments", patience,
                )
                break

            # Propose
            strategy_fn = _select_strategy()
            proposal = strategy_fn(self.state)

            logger.info(
                "[%d/%d] Strategy: %s | algo=%s | features=%d",
                i, max_experiments, proposal["strategy"],
                proposal["algorithm"], len(proposal["feature_set"]),
            )

            # Train & evaluate
            metrics, duration = _run_single_experiment(
                self.db, proposal, self.target, self.split_config,
            )

            if metrics is None or self.objective not in metrics:
                logger.warning("[%d/%d] Failed or missing metric, skipping", i, max_experiments)
                self._log_experiment(i, proposal, metrics, duration, accepted=False)
                no_improvement_count += 1
                continue

            score = metrics[self.objective]
            accepted = _is_improvement(score, self.state["best_score"], self.higher_is_better)

            if accepted:
                old_score = self.state["best_score"]
                self.state["best_score"] = score
                self.state["best_algorithm"] = proposal["algorithm"]
                self.state["best_hyperparams"] = proposal["hyperparameters"]
                self.state["best_features"] = proposal["feature_set"]
                no_improvement_count = 0

                logger.info(
                    "[%d/%d] IMPROVEMENT: %s %.4f -> %.4f (%s)",
                    i, max_experiments, self.objective, old_score, score,
                    proposal["strategy"],
                )

                if on_improvement:
                    on_improvement(i, score, proposal)
            else:
                no_improvement_count += 1
                logger.info(
                    "[%d/%d] No improvement: %.4f vs best %.4f (patience %d/%d)",
                    i, max_experiments, score, self.state["best_score"],
                    no_improvement_count, patience,
                )

            self._log_experiment(i, proposal, metrics, duration, accepted)
            self._save_experiment(
                proposal, metrics, duration, is_best=accepted, trial_num=i,
            )

        summary = self._build_summary()
        logger.info(
            "Autoresearch complete: %d experiments, %d improvements, best %s = %.4f",
            len(self.history),
            sum(1 for h in self.history if h["accepted"]),
            self.objective,
            self.state["best_score"],
        )
        return summary

    def _log_experiment(
        self,
        trial_num: int,
        proposal: dict,
        metrics: dict | None,
        duration: float | None,
        accepted: bool,
    ) -> None:
        """Append to in-memory history."""
        self.history.append({
            "trial": trial_num,
            "strategy": proposal.get("strategy", "unknown"),
            "algorithm": proposal["algorithm"],
            "hyperparameters": proposal["hyperparameters"],
            "feature_set": proposal["feature_set"],
            "score": metrics.get(self.objective) if metrics else None,
            "metrics": metrics,
            "duration_s": duration,
            "accepted": accepted,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _save_experiment(
        self,
        proposal: dict,
        metrics: dict | None,
        duration: float | None,
        is_best: bool,
        trial_num: int,
    ) -> None:
        """Save experiment to DB for traceability."""
        label = "BEST" if is_best else "trial"
        experiment = Experiment(
            name=f"autoresearch_{trial_num:04d}_{proposal.get('strategy', 'unknown')}",
            description=(
                f"Autoresearch trial {trial_num} | "
                f"strategy={proposal.get('strategy')} | "
                f"objective={self.objective} | "
                f"{'ACCEPTED' if is_best else 'rejected'}"
            ),
            algorithm=proposal["algorithm"],
            target=self.target,
            hyperparameters=proposal["hyperparameters"],
            feature_set=proposal["feature_set"],
            split_config=self.split_config,
            status="completed" if metrics else "failed",
            metrics=metrics,
            training_duration_s=duration,
            completed_at=datetime.utcnow(),
        )
        self.db.add(experiment)
        self.db.commit()

    def _build_summary(self) -> dict[str, Any]:
        """Build a summary of the autoresearch run."""
        improvements = [h for h in self.history if h["accepted"]]
        strategy_counts: dict[str, int] = {}
        strategy_improvements: dict[str, int] = {}
        for h in self.history:
            s = h["strategy"]
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
            if h["accepted"]:
                strategy_improvements[s] = strategy_improvements.get(s, 0) + 1

        return {
            "objective": self.objective,
            "direction": "maximize" if self.higher_is_better else "minimize",
            "best_score": self.state["best_score"],
            "best_algorithm": self.state["best_algorithm"],
            "best_hyperparameters": self.state["best_hyperparams"],
            "best_feature_set": self.state["best_features"],
            "total_experiments": len(self.history),
            "total_improvements": len(improvements),
            "improvement_rate": (
                round(len(improvements) / max(len(self.history), 1) * 100, 1)
            ),
            "total_duration_s": round(
                sum(h["duration_s"] for h in self.history if h["duration_s"]), 1,
            ),
            "strategy_counts": strategy_counts,
            "strategy_improvements": strategy_improvements,
            "score_trajectory": [
                {"trial": h["trial"], "score": h["score"], "strategy": h["strategy"]}
                for h in improvements
            ],
            "history": self.history,
        }
