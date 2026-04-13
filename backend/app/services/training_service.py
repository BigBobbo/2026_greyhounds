"""
Training service: orchestrates the full ML training pipeline.

1. Materialize features
2. Build dataset
3. Train model
4. Evaluate
5. Calibrate probabilities (isotonic regression)
6. Save model + metrics to DB
"""

import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
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


class _TrainingLogHandler(logging.Handler):
    """Captures log records and auto-flushes to DB every few seconds."""

    FLUSH_INTERVAL = 5  # seconds

    def __init__(self, db: Session, experiment: "Experiment") -> None:
        super().__init__(level=logging.DEBUG)
        self.buffer: list[str] = []
        self.formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._db = db
        self._experiment = experiment
        self._last_flush = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(self.format(record))
            now = time.time()
            if now - self._last_flush >= self.FLUSH_INTERVAL:
                self._flush_to_db()
                self._last_flush = now
        except Exception:
            # Don't let logging failures crash the training pipeline,
            # but record the issue so it's diagnosable.
            import sys
            print(f"[TrainingLogHandler] emit failed: {sys.exc_info()[1]}", file=sys.stderr)

    def _flush_to_db(self) -> None:
        """Write the current log buffer to the experiment row."""
        try:
            self._experiment.training_log = self.get_log_text()
            self._experiment.heartbeat_at = datetime.utcnow()
            self._db.commit()
        except Exception:
            import sys
            print(f"[TrainingLogHandler] DB flush failed: {sys.exc_info()[1]}", file=sys.stderr)
            try:
                self._db.rollback()
            except Exception:
                pass

    def get_log_text(self) -> str:
        return "\n".join(self.buffer)


def _heartbeat(db: Session, experiment: "Experiment", stage: str,
               log_handler: _TrainingLogHandler | None = None) -> None:
    """Update the heartbeat timestamp, training stage, and flush log."""
    experiment.heartbeat_at = datetime.utcnow()
    experiment.training_stage = stage
    if log_handler is not None:
        experiment.training_log = log_handler.get_log_text()
    db.commit()


def create_trainer(algorithm: str, params: dict[str, Any], target: str):
    """Factory function to create the appropriate trainer."""
    target_type = "regression" if target == "finish_time" else "classification"

    if algorithm == "lambdarank":
        from ml.trainers.lambdarank_trainer import LambdaRankTrainer
        return LambdaRankTrainer(params, target_type)
    elif algorithm == "xgboost":
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
    experiment.heartbeat_at = datetime.utcnow()
    experiment.training_stage = "starting"
    experiment.training_log = ""
    db.commit()

    # Attach a log handler that auto-flushes to DB every 5s
    log_handler = _TrainingLogHandler(db, experiment)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    start_time = time.time()

    try:
        # 1. Build dataset
        logger.info("Building dataset for experiment %d", experiment_id)
        _heartbeat(db, experiment, "building_dataset", log_handler)
        split_cfg = experiment.split_config or {}
        # LambdaRank always uses finish_position internally
        build_target = "finish_position" if experiment.algorithm == "lambdarank" else experiment.target
        dataset = build_dataset(
            db,
            feature_ids=experiment.feature_set,
            target=build_target,
            split_config=split_cfg,
            only_complete=split_cfg.get("only_complete", False),
            version_id=split_cfg.get("version_id"),
        )

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]
        feature_names = dataset["feature_names"]

        # Persist the actual cutoff dates so prediction service can guard against leakage
        split_config = dict(experiment.split_config or {})
        split_config["train_cutoff_date"] = dataset["stats"].get("train_cutoff_date")
        split_config["test_cutoff_date"] = dataset["stats"].get("test_cutoff_date")
        experiment.split_config = split_config
        db.commit()

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

        is_ranking = experiment.algorithm == "lambdarank"
        group_train = dataset.get("group_train")
        group_val = dataset.get("group_val")
        group_test = dataset.get("group_test")

        # 3. Train
        logger.info("Training %s model...", experiment.algorithm)
        _heartbeat(db, experiment, "training_model", log_handler)
        if is_ranking:
            result = trainer.train(X_train, y_train, X_val, y_val,
                                   group_train=group_train, group_val=group_val)
        else:
            result = trainer.train(X_train, y_train, X_val, y_val)

        # 4. Evaluate on test set
        logger.info("Evaluating on test set...")
        _heartbeat(db, experiment, "evaluating", log_handler)
        from ml.evaluation import compute_metrics, compute_betting_metrics
        target_type = "regression" if experiment.target == "finish_time" else "classification"

        test_pred = trainer.predict(X_test)

        if is_ranking:
            # For ranking models, convert scores to probabilities via softmax
            test_proba = trainer.scores_to_proba(test_pred, group_test)
            # Compute ranking-specific metrics
            test_metrics = trainer._compute_ranking_metrics(y_test, test_pred, group_test)
            # Also compute classification metrics using win labels
            y_test_binary = (y_test == 1).astype(float)
            test_pred_binary = np.zeros_like(y_test_binary)
            # Mark top pick per race as predicted winner
            idx = 0
            for g_size in (group_test or [len(test_pred)]):
                g_scores = test_pred[idx:idx + g_size]
                winner_idx = np.argmax(g_scores)
                test_pred_binary[idx + winner_idx] = 1
                idx += g_size
            cls_metrics = compute_metrics(y_test_binary, test_pred_binary, test_proba, "classification")
            test_metrics.update(cls_metrics)
        else:
            test_proba = trainer.predict_proba(X_test)
            test_metrics = compute_metrics(y_test, test_pred, test_proba, target_type)

        # Merge val and test metrics
        all_metrics = {f"val_{k}": v for k, v in result.metrics.items()}
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

        # 4b. Betting P&L evaluation
        meta_test = dataset.get("meta_test")
        betting_data = None
        can_eval_betting = (
            (target_type == "classification" or is_ranking)
            and test_proba is not None
            and meta_test is not None
        )
        if can_eval_betting:
            try:
                y_binary = (y_test == 1).astype(float).values if is_ranking else y_test.values
                betting = compute_betting_metrics(
                    y_binary,
                    test_proba,
                    meta_test["sp_decimal"].values,
                    meta_test["race_id"].values,
                )
                betting_data = betting
                # Add headline betting metrics
                all_metrics["betting_top_pick_pnl"] = betting["top_pick_pnl"]
                all_metrics["betting_top_pick_roi"] = betting["top_pick_roi"]
                all_metrics["betting_top_pick_strike_rate"] = betting["top_pick_strike_rate"]
                all_metrics["betting_value_pnl"] = betting["value_bet_pnl"]
                all_metrics["betting_value_roi"] = betting["value_bet_roi"]
                all_metrics["betting_favourite_pnl"] = betting["favourite_pnl"]
                all_metrics["betting_favourite_roi"] = betting["favourite_roi"]
                all_metrics["betting_kelly_pnl"] = betting.get("kelly_pnl", 0)
                all_metrics["betting_kelly_roi"] = betting.get("kelly_roi", 0)
                logger.info(
                    "Betting metrics: top_pick_pnl=$%.2f (ROI %.1f%%), value_pnl=$%.2f, kelly_pnl=$%.2f",
                    betting["top_pick_pnl"], betting["top_pick_roi"],
                    betting["value_bet_pnl"], betting.get("kelly_pnl", 0),
                )
            except Exception as e:
                logger.warning("Betting metrics failed: %s", e)

        # 5. Fit probability calibrator on validation set (isotonic regression)
        _heartbeat(db, experiment, "calibrating", log_handler)
        calibrator = None
        test_proba_calibrated = None
        if (target_type == "classification" or is_ranking) and test_proba is not None:
            try:
                if is_ranking:
                    val_scores = trainer.predict(X_val)
                    val_proba = trainer.scores_to_proba(val_scores, group_val)
                    y_val_binary = (y_val == 1).astype(float).values
                else:
                    val_proba = trainer.predict_proba(X_val)
                    y_val_binary = y_val.values

                if val_proba is not None and len(val_proba) > 10:
                    calibrator = IsotonicRegression(
                        y_min=0.01, y_max=0.99, out_of_bounds="clip",
                    )
                    calibrator.fit(val_proba, y_val_binary)

                    # Re-calibrate test probabilities for evaluation
                    test_proba_calibrated = calibrator.predict(test_proba)
                    logger.info(
                        "Calibrator fitted. Test proba range: [%.3f, %.3f] -> calibrated: [%.3f, %.3f]",
                        test_proba.min(), test_proba.max(),
                        test_proba_calibrated.min(), test_proba_calibrated.max(),
                    )
                    # Add calibrated metrics alongside raw metrics
                    all_metrics["calibrated_brier_score"] = float(
                        np.mean((test_proba_calibrated - (y_test_binary if is_ranking else y_test.values)) ** 2)
                    )
                    all_metrics["raw_brier_score"] = float(
                        np.mean((test_proba - (y_test_binary if is_ranking else y_test.values)) ** 2)
                    )
            except Exception as e:
                logger.warning("Calibration fitting failed: %s", e)

        # 6. Additional evaluation data
        _heartbeat(db, experiment, "computing_shap", log_handler)
        confusion = None
        roc_data = None
        calibration = None
        shap_data = None

        if target_type == "classification" or is_ranking:
            if is_ranking:
                y_binary = (y_test == 1).astype(float).values
                confusion = compute_confusion_matrix(y_binary, test_pred_binary)
            else:
                y_binary = y_test.values
                confusion = compute_confusion_matrix(y_binary, test_pred)

            if test_proba is not None:
                # Use calibrated probabilities for evaluation if available
                eval_proba = test_proba_calibrated if calibrator is not None else test_proba
                roc_data = compute_roc_data(y_binary, eval_proba)
                calibration = compute_calibration_data(y_binary, eval_proba)

        # SHAP (skip if too slow — limit samples)
        try:
            shap_data = compute_shap_summary(
                trainer.model, X_test, feature_names, max_samples=300,
            )
        except Exception as e:
            logger.warning("SHAP failed: %s", e)

        # 7. Save model + preprocessing artifacts
        _heartbeat(db, experiment, "saving_model", log_handler)
        model_dir = settings.model_artifacts_dir
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"experiment_{experiment_id}.joblib")
        # Save the trainer along with feature medians and calibrator
        artifact = {
            "trainer": trainer,
            "feature_medians": dataset.get("feature_medians", {}),
            "is_ranking": is_ranking,
            "calibrator": calibrator,
        }
        joblib.dump(artifact, model_path)

        # 8. Update experiment record
        duration = time.time() - start_time
        logger.info(
            "Experiment %d completed in %.1fs. Test metrics: %s",
            experiment_id, duration, test_metrics,
        )
        experiment.status = "completed"
        experiment.heartbeat_at = None
        experiment.training_stage = None
        experiment.metrics = all_metrics
        experiment.confusion_matrix = confusion
        experiment.roc_data = roc_data
        experiment.calibration_data = {
            "calibration": calibration,
            "betting": betting_data,
        } if calibration or betting_data else None
        experiment.shap_summary = shap_data
        experiment.feature_importance = result.feature_importance
        experiment.training_duration_s = duration
        experiment.model_path = model_path
        experiment.completed_at = datetime.utcnow()
        experiment.training_log = log_handler.get_log_text()
        db.commit()

    except Exception as e:
        tb = traceback.format_exc()
        # Append error details directly to the log buffer before any DB
        # operations so the traceback is preserved even if the DB commit
        # below fails (which would leave the experiment stuck in "running"
        # with no error details visible).
        log_handler.buffer.append(
            f"--- TRAINING FAILED ---\n{type(e).__name__}: {e}\n{tb}"
        )
        logger.error("Experiment %d failed: %s", experiment_id, e, exc_info=True)
        try:
            db.rollback()
        except Exception:
            pass
        try:
            experiment.status = "failed"
            experiment.error_message = f"{type(e).__name__}: {e}\n\n{tb}"
            experiment.training_duration_s = time.time() - start_time
            experiment.completed_at = datetime.utcnow()
            experiment.heartbeat_at = None
            experiment.training_stage = None
            experiment.training_log = log_handler.get_log_text()
            db.commit()
        except Exception as commit_err:
            logger.error("Failed to persist error for experiment %d: %s", experiment_id, commit_err)
            # Last-resort: try with a fresh connection so the error isn't lost
            try:
                db.rollback()
                experiment.status = "failed"
                experiment.error_message = f"{type(e).__name__}: {e}\n\n{tb}"
                experiment.training_log = log_handler.get_log_text()
                experiment.heartbeat_at = None
                experiment.training_stage = None
                db.commit()
            except Exception:
                import sys
                print(
                    f"[TrainingService] CRITICAL: Could not persist error for "
                    f"experiment {experiment_id}: {e}\n{tb}",
                    file=sys.stderr,
                )
    finally:
        root_logger.removeHandler(log_handler)


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
    experiment.heartbeat_at = datetime.utcnow()
    experiment.training_stage = "starting"
    experiment.training_log = ""
    db.commit()

    log_handler = _TrainingLogHandler(db, experiment)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    start_time = time.time()

    try:
        logger.info("Building dataset for Optuna experiment %d (%d trials)", experiment_id, n_trials)
        _heartbeat(db, experiment, "building_dataset", log_handler)
        split_cfg = experiment.split_config or {}
        build_target = "finish_position" if experiment.algorithm == "lambdarank" else experiment.target
        dataset = build_dataset(
            db,
            feature_ids=experiment.feature_set,
            target=build_target,
            split_config=split_cfg,
            only_complete=split_cfg.get("only_complete", False),
            version_id=split_cfg.get("version_id"),
        )

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_val = dataset["X_val"]
        y_val = dataset["y_val"]
        feature_names = dataset["feature_names"]
        target_type = "regression" if experiment.target == "finish_time" else "classification"

        # Persist the actual cutoff dates so prediction service can guard against leakage
        split_config = dict(experiment.split_config or {})
        split_config["train_cutoff_date"] = dataset["stats"].get("train_cutoff_date")
        split_config["test_cutoff_date"] = dataset["stats"].get("test_cutoff_date")
        experiment.split_config = split_config
        db.commit()

        is_ranking = experiment.algorithm == "lambdarank"
        group_train = dataset.get("group_train")
        group_val = dataset.get("group_val")

        def objective(trial):
            _heartbeat(db, experiment, f"optuna_trial_{trial.number + 1}_of_{n_trials}", log_handler)
            params = _suggest_params(trial, experiment.algorithm)
            trainer = create_trainer(experiment.algorithm, params, experiment.target)
            if is_ranking:
                result = trainer.train(X_train, y_train, X_val, y_val,
                                       group_train=group_train, group_val=group_val)
                # Optimize for top-1 accuracy (maximize), return negative
                return -(result.metrics.get("top1_accuracy", 0))
            elif target_type == "classification":
                result = trainer.train(X_train, y_train, X_val, y_val)
                return result.metrics.get("log_loss", 999)
            else:
                result = trainer.train(X_train, y_train, X_val, y_val)
                return result.metrics.get("rmse", 999)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Retrain with best params
        _heartbeat(db, experiment, "retraining_best", log_handler)
        best_params = study.best_params
        experiment.hyperparameters = best_params

        trainer = create_trainer(experiment.algorithm, best_params, experiment.target)
        if is_ranking:
            result = trainer.train(X_train, y_train, X_val, y_val,
                                   group_train=group_train, group_val=group_val)
        else:
            result = trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate on test
        _heartbeat(db, experiment, "evaluating", log_handler)
        X_test, y_test = dataset["X_test"], dataset["y_test"]
        group_test = dataset.get("group_test")
        from ml.evaluation import compute_metrics
        test_pred = trainer.predict(X_test)

        if is_ranking:
            test_proba = trainer.scores_to_proba(test_pred, group_test)
            test_metrics = trainer._compute_ranking_metrics(y_test, test_pred, group_test)
            y_binary = (y_test == 1).astype(float)
            cls_metrics = compute_metrics(y_binary, (test_proba > 0.5).astype(float), test_proba, "classification")
            test_metrics.update(cls_metrics)
        else:
            test_proba = trainer.predict_proba(X_test)
            test_metrics = compute_metrics(y_test, test_pred, test_proba, target_type)

        all_metrics = {f"val_{k}": v for k, v in result.metrics.items()}
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        all_metrics["optuna_best_value"] = float(study.best_value)
        all_metrics["optuna_n_trials"] = n_trials

        # Betting P&L evaluation
        meta_test = dataset.get("meta_test")
        betting_data = None
        can_eval_betting = (
            (target_type == "classification" or is_ranking)
            and test_proba is not None
            and meta_test is not None
        )
        if can_eval_betting:
            try:
                from ml.evaluation import compute_betting_metrics
                y_binary_bet = (y_test == 1).astype(float).values if is_ranking else y_test.values
                betting = compute_betting_metrics(
                    y_binary_bet,
                    test_proba,
                    meta_test["sp_decimal"].values,
                    meta_test["race_id"].values,
                )
                betting_data = betting
                all_metrics["betting_top_pick_pnl"] = betting["top_pick_pnl"]
                all_metrics["betting_top_pick_roi"] = betting["top_pick_roi"]
                all_metrics["betting_top_pick_strike_rate"] = betting["top_pick_strike_rate"]
                all_metrics["betting_value_pnl"] = betting["value_bet_pnl"]
                all_metrics["betting_value_roi"] = betting["value_bet_roi"]
                all_metrics["betting_favourite_pnl"] = betting["favourite_pnl"]
                all_metrics["betting_favourite_roi"] = betting["favourite_roi"]
                all_metrics["betting_kelly_pnl"] = betting.get("kelly_pnl", 0)
                all_metrics["betting_kelly_roi"] = betting.get("kelly_roi", 0)
            except Exception as e:
                logger.warning("Optuna betting metrics failed: %s", e)

        # Fit calibrator on validation set
        calibrator = None
        if (target_type == "classification" or is_ranking) and test_proba is not None:
            try:
                if is_ranking:
                    val_scores_cal = trainer.predict(X_val)
                    val_proba_cal = trainer.scores_to_proba(val_scores_cal, group_val)
                    y_val_binary = (y_val == 1).astype(float).values
                else:
                    val_proba_cal = trainer.predict_proba(X_val)
                    y_val_binary = y_val.values

                if val_proba_cal is not None and len(val_proba_cal) > 10:
                    calibrator = IsotonicRegression(
                        y_min=0.01, y_max=0.99, out_of_bounds="clip",
                    )
                    calibrator.fit(val_proba_cal, y_val_binary)
            except Exception as e:
                logger.warning("Optuna calibration fitting failed: %s", e)

        # SHAP analysis (same as standard training path)
        shap_data = None
        try:
            shap_data = compute_shap_summary(
                trainer.model, X_test, feature_names, max_samples=300,
            )
        except Exception as e:
            logger.warning("SHAP failed: %s", e)

        # Save model + preprocessing artifacts
        _heartbeat(db, experiment, "saving_model", log_handler)
        model_dir = settings.model_artifacts_dir
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"experiment_{experiment_id}.joblib")
        artifact = {
            "trainer": trainer,
            "feature_medians": dataset.get("feature_medians", {}),
            "is_ranking": is_ranking,
            "calibrator": calibrator,
        }
        joblib.dump(artifact, model_path)

        logger.info("Optuna experiment %d completed. Best: %s", experiment_id, best_params)

        experiment.status = "completed"
        experiment.heartbeat_at = None
        experiment.training_stage = None
        experiment.metrics = all_metrics
        experiment.feature_importance = result.feature_importance
        experiment.shap_summary = shap_data
        experiment.training_duration_s = time.time() - start_time
        experiment.model_path = model_path
        experiment.completed_at = datetime.utcnow()

        if (target_type == "classification" or is_ranking) and test_proba is not None:
            y_binary = (y_test == 1).astype(float).values if is_ranking else y_test.values
            pred_binary = (test_proba > 0.5).astype(float) if is_ranking else test_pred
            experiment.confusion_matrix = compute_confusion_matrix(y_binary, pred_binary)
            # Use calibrated probabilities if available
            eval_proba = calibrator.predict(test_proba) if calibrator is not None else test_proba
            experiment.roc_data = compute_roc_data(y_binary, eval_proba)
            calibration = compute_calibration_data(y_binary, eval_proba)
            experiment.calibration_data = {
                "calibration": calibration,
                "betting": betting_data,
            } if calibration or betting_data else None

        experiment.training_log = log_handler.get_log_text()
        db.commit()

    except Exception as e:
        tb = traceback.format_exc()
        log_handler.buffer.append(
            f"--- TRAINING FAILED ---\n{type(e).__name__}: {e}\n{tb}"
        )
        logger.error("Optuna experiment %d failed: %s", experiment_id, e, exc_info=True)
        try:
            db.rollback()
        except Exception:
            pass
        try:
            experiment.status = "failed"
            experiment.error_message = f"{type(e).__name__}: {e}\n\n{tb}"
            experiment.completed_at = datetime.utcnow()
            experiment.heartbeat_at = None
            experiment.training_stage = None
            experiment.training_log = log_handler.get_log_text()
            db.commit()
        except Exception as commit_err:
            logger.error("Failed to persist error for Optuna experiment %d: %s", experiment_id, commit_err)
            try:
                db.rollback()
                experiment.status = "failed"
                experiment.error_message = f"{type(e).__name__}: {e}\n\n{tb}"
                experiment.training_log = log_handler.get_log_text()
                experiment.heartbeat_at = None
                experiment.training_stage = None
                db.commit()
            except Exception:
                import sys
                print(
                    f"[TrainingService] CRITICAL: Could not persist error for "
                    f"Optuna experiment {experiment_id}: {e}\n{tb}",
                    file=sys.stderr,
                )
    finally:
        root_logger.removeHandler(log_handler)


def _suggest_params(trial, algorithm: str) -> dict:
    """Suggest hyperparameters for Optuna trial."""
    if algorithm == "lambdarank":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
    elif algorithm == "xgboost":
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
