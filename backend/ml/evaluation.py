"""
Model evaluation: metrics, confusion matrix, ROC, calibration data.

Returns all data as JSON-serializable dicts/lists for storage in the DB
and rendering in the frontend.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    task_type: str,
) -> dict[str, float]:
    """Compute evaluation metrics based on task type."""
    metrics: dict[str, float] = {}

    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        if y_proba is not None:
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            except Exception:
                pass
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                pass
            try:
                metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))
            except Exception:
                pass

        # Win prediction specific
        if len(np.unique(y_true)) == 2:
            winners_predicted = np.sum((y_pred == 1) & (y_true == 1))
            total_winners = np.sum(y_true == 1)
            if total_winners > 0:
                metrics["winner_recall"] = float(winners_predicted / total_winners)
            metrics["winner_precision"] = float(
                winners_predicted / max(np.sum(y_pred == 1), 1)
            )

    elif task_type == "regression":
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["median_ae"] = float(np.median(np.abs(y_true - y_pred)))

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Compute confusion matrix as a nested list."""
    cm = confusion_matrix(y_true, y_pred)
    return cm.tolist()


def compute_roc_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, list[float]]:
    """Compute ROC curve data points."""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        # Downsample for storage (max 200 points)
        if len(fpr) > 200:
            indices = np.linspace(0, len(fpr) - 1, 200, dtype=int)
            fpr, tpr = fpr[indices], tpr[indices]
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
    except Exception:
        return {"fpr": [], "tpr": []}


def compute_calibration_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Compute calibration curve data (predicted prob vs actual frequency)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_trues = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(float(y_proba[mask].mean()))
            bin_trues.append(float(y_true[mask].mean()))
            bin_counts.append(int(mask.sum()))

    return {
        "predicted_prob": bin_means,
        "actual_freq": bin_trues,
        "bin_counts": bin_counts,
    }


def compute_shap_summary(model: Any, X: np.ndarray, feature_names: list[str], max_samples: int = 500) -> dict[str, Any] | None:
    """Compute SHAP values summary. Returns mean absolute SHAP values per feature."""
    try:
        import shap

        if X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
        else:
            X_sample = X

        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, X_sample[:100])

        shap_values = explainer.shap_values(X_sample)

        # Handle multi-output (classification returns list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs = np.abs(shap_values).mean(axis=0)
        feature_shap = dict(zip(feature_names, mean_abs.tolist()))

        # Sort by importance
        feature_shap = dict(sorted(feature_shap.items(), key=lambda x: x[1], reverse=True))

        return {"mean_abs_shap": feature_shap}

    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None
