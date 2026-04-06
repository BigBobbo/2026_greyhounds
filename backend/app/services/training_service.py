"""
Training service: orchestrates the full ML training pipeline.

1. Materialize features
2. Build dataset
3. Train model
4. Evaluate
5. Save model + metrics to DB
"""

import logging
import os
import time
from datetime import datetime
from typing import Any

import joblib
from sqlalchemy.orm import Session

from app.config import settings
from app.models.experiment import Experiment
from ml.dataset_builder import build_dataset
from ml.evaluation import (
    compute_calibration_data,
    compute_confusion_matrix,
    compute_roc_data,
    compute_shap_summary,
)

logger = logging.getLogger(__name__)


def create_trainer(algorithm: str, params: dict[str, Any], target: str):
    """Factory function to create the appropriate trainer."""
    target_type = "regression" if target == "finish_time" else "classification"

    if algorithm == "xgboost":
        from ml.trainers.xgboost_trainer import XGBoostTrainer
        return XGBoostTrainer(params, target_type)
    elif algorithm == "lightgbm":
        from ml.trainers.lightgbm_trainer import LightGBMTrainer
        return LightGBMTrainer(params, target_type)
    elif algorithm in ("logistic_regression", "random_forest"):
        from ml.trainers.sklearn_trainer import SklearnTrainer
        return SklearnTrainer(params, algorithm, target_type)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_training(db: Session, experiment_id: int) -> None:
    """Run the full training pipeline for an experiment."""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        logger.error("Experiment %d not found", experiment_id)
        return

    experiment.status = "running"
    db.commit()

    start_time = time.time()

    try:
        # 1. Build dataset
        logger.info("Building dataset for experiment %d", experiment_id)
        dataset = build_dataset(
            db,
            feature_ids=experiment.feature_set,
            target=experiment.target,
            split_config=experiment.split_config,
        )

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]
        feature_names = dataset["feature_names"]

        logger.info(
            "Dataset built: train=%d, val=%d, test=%d, features=%d",
            len(X_train), len(X_val), len(X_test), len(feature_names),
        )

        # 2. Create trainer
        trainer = create_trainer(
            experiment.algorithm,
            experiment.hyperparameters,
            experiment.target,
        )

        # 3. Train
        logger.info("Training %s model...", experiment.algorithm)
        result = trainer.train(X_train, y_train, X_val, y_val)

        # 4. Evaluate on test set
        logger.info("Evaluating on test set...")
        from ml.evaluation import compute_metrics
        target_type = "regression" if experiment.target == "finish_time" else "classification"

        test_pred = trainer.predict(X_test)
        test_proba = trainer.predict_proba(X_test)
        test_metrics = compute_metrics(y_test, test_pred, test_proba, target_type)

        # Merge val and test metrics
        all_metrics = {f"val_{k}": v for k, v in result.metrics.items()}
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        # 5. Additional evaluation data
        confusion = None
        roc_data = None
        calibration = None
        shap_data = None

        if target_type == "classification":
            confusion = compute_confusion_matrix(y_test.values, test_pred)

            if test_proba is not None:
                roc_data = compute_roc_data(y_test.values, test_proba)
                calibration = compute_calibration_data(y_test.values, test_proba)

        # SHAP (skip if too slow — limit samples)
        try:
            shap_data = compute_shap_summary(
                trainer.model, X_test, feature_names, max_samples=300,
            )
        except Exception as e:
            logger.warning("SHAP failed: %s", e)

        # 6. Save model
        model_dir = settings.model_artifacts_dir
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"experiment_{experiment_id}.joblib")
        joblib.dump(trainer, model_path)

        # 7. Update experiment record
        duration = time.time() - start_time
        experiment.status = "completed"
        experiment.metrics = all_metrics
        experiment.confusion_matrix = confusion
        experiment.roc_data = roc_data
        experiment.calibration_data = calibration
        experiment.shap_summary = shap_data
        experiment.feature_importance = result.feature_importance
        experiment.training_duration_s = duration
        experiment.model_path = model_path
        experiment.completed_at = datetime.utcnow()
        db.commit()

        logger.info(
            "Experiment %d completed in %.1fs. Test metrics: %s",
            experiment_id, duration, test_metrics,
        )

    except Exception as e:
        logger.error("Experiment %d failed: %s", experiment_id, e, exc_info=True)
        experiment.status = "failed"
        experiment.error_message = f"{type(e).__name__}: {e}"
        experiment.training_duration_s = time.time() - start_time
        experiment.completed_at = datetime.utcnow()
        db.commit()


def run_optuna_optimization(
    db: Session,
    experiment_id: int,
    n_trials: int = 50,
) -> None:
    """Run Optuna hyperparameter optimization."""
    import optuna

    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        return

    experiment.status = "running"
    db.commit()

    start_time = time.time()

    try:
        dataset = build_dataset(
            db,
            feature_ids=experiment.feature_set,
            target=experiment.target,
            split_config=experiment.split_config,
        )

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        target_type = "regression" if experiment.target == "finish_time" else "classification"

        def objective(trial):
            params = _suggest_params(trial, experiment.algorithm)
            trainer = create_trainer(experiment.algorithm, params, experiment.target)
            result = trainer.train(X_train, y_train, X_val, y_val)

            if target_type == "classification":
                return result.metrics.get("log_loss", 999)
            else:
                return result.metrics.get("rmse", 999)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Retrain with best params
        best_params = study.best_params
        experiment.hyperparameters = best_params

        trainer = create_trainer(experiment.algorithm, best_params, experiment.target)
        result = trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate on test
        X_test, y_test = dataset["X_test"], dataset["y_test"]
        from ml.evaluation import compute_metrics
        test_pred = trainer.predict(X_test)
        test_proba = trainer.predict_proba(X_test)
        test_metrics = compute_metrics(y_test, test_pred, test_proba, target_type)

        all_metrics = {f"val_{k}": v for k, v in result.metrics.items()}
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        all_metrics["optuna_best_value"] = float(study.best_value)
        all_metrics["optuna_n_trials"] = n_trials

        # Save
        model_dir = settings.model_artifacts_dir
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"experiment_{experiment_id}.joblib")
        joblib.dump(trainer, model_path)

        experiment.status = "completed"
        experiment.metrics = all_metrics
        experiment.feature_importance = result.feature_importance
        experiment.training_duration_s = time.time() - start_time
        experiment.model_path = model_path
        experiment.completed_at = datetime.utcnow()

        if target_type == "classification" and test_proba is not None:
            experiment.confusion_matrix = compute_confusion_matrix(y_test.values, test_pred)
            experiment.roc_data = compute_roc_data(y_test.values, test_proba)
            experiment.calibration_data = compute_calibration_data(y_test.values, test_proba)

        db.commit()
        logger.info("Optuna experiment %d completed. Best: %s", experiment_id, best_params)

    except Exception as e:
        logger.error("Optuna experiment %d failed: %s", experiment_id, e, exc_info=True)
        experiment.status = "failed"
        experiment.error_message = str(e)
        experiment.completed_at = datetime.utcnow()
        db.commit()


def _suggest_params(trial, algorithm: str) -> dict:
    """Suggest hyperparameters for Optuna trial."""
    if algorithm == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
    elif algorithm == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
    elif algorithm == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif algorithm == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 100, log=True),
        }
    return {}
