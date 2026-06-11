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


def normalize_probs_per_race(y_proba: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Normalize win probabilities to sum to 1 within each race.

    This is the convention the serving path uses for every model type, so
    any evaluation of betting strategies must be computed on the same
    numbers — otherwise the backtest measures a quantity that is never bet
    with (e.g. a race whose calibrated probs sum to 0.85 gets every
    probability inflated ~18% at serve time, manufacturing edge the
    backtest never validated). Division by the per-race sum is monotonic,
    so rankings/top picks are unchanged.
    """
    proba = np.asarray(y_proba, dtype=float).copy()
    ids = np.asarray(race_ids)
    for rid in np.unique(ids):
        mask = ids == rid
        total = proba[mask].sum()
        if total > 0:
            proba[mask] = proba[mask] / total
    return proba


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


def compute_betting_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sp_decimal: np.ndarray,
    race_ids: np.ndarray,
) -> dict[str, Any]:
    """
    Compute betting P&L metrics.

    Simulates betting $1 on the model's top pick in each race.
    Also evaluates value betting (bet when model prob > implied prob).

    Returns:
        {
            "top_pick_pnl": total P&L from $1 on predicted winner per race,
            "top_pick_roi": return on investment %,
            "top_pick_races": number of races bet on,
            "top_pick_winners": number of winners picked,
            "top_pick_strike_rate": % of races where top pick won,
            "value_bet_pnl": P&L from $1 on value bets only,
            "value_bet_roi": ROI % for value bets,
            "value_bet_count": number of value bets placed,
            "value_bet_winners": number of value bet winners,
            "favourite_pnl": P&L from $1 on SP favourite (baseline),
            "favourite_roi": ROI % for favourites,
            "pnl_by_race": list of per-race results for charting,
        }
    """
    import pandas as pd

    # Match the serving convention BEFORE any filtering: normalize over the
    # full field of each race, then drop no-SP rows (serving normalizes over
    # all runners regardless of SP availability).
    y_proba = normalize_probs_per_race(y_proba, race_ids)

    df = pd.DataFrame({
        "won": y_true.astype(bool),
        "prob": y_proba,
        "sp": sp_decimal,
        "race_id": race_ids,
    })

    # Drop rows with no SP
    df = df[df["sp"].notna() & (df["sp"] > 1)]

    if df.empty:
        return {"error": "No entries with SP data"}

    # Implied probability from SP
    df["implied_prob"] = 1.0 / df["sp"]

    # --- Strategy 1: Bet $1 on model's top pick per race ---
    # Also compute favourite (lowest SP) baseline in the same loop for alignment
    top_pick_results = []
    fav_profits = []
    for race_id, group in df.groupby("race_id"):
        if len(group) == 0:
            continue
        # Model's top pick = highest predicted probability
        top = group.loc[group["prob"].idxmax()]
        profit = (top["sp"] - 1) if top["won"] else -1.0
        top_pick_results.append({
            "race_id": int(race_id),
            "won": bool(top["won"]),
            "sp": float(top["sp"]),
            "prob": float(top["prob"]),
            "profit": float(profit),
        })
        # Favourite = lowest SP (baseline)
        fav = group.loc[group["sp"].idxmin()]
        fav_profit = (fav["sp"] - 1) if fav["won"] else -1.0
        fav_profits.append(float(fav_profit))

    top_pick_df = pd.DataFrame(top_pick_results)
    top_pick_pnl = float(top_pick_df["profit"].sum()) if len(top_pick_df) > 0 else 0
    top_pick_races = len(top_pick_df)
    top_pick_winners = int(top_pick_df["won"].sum()) if len(top_pick_df) > 0 else 0

    # --- Strategy 2: Value betting (model prob > implied prob * 1.05 = 5% min edge) ---
    min_edge = 0.05
    df["edge"] = df["prob"] - df["implied_prob"]
    df["is_value"] = df["edge"] > min_edge
    value_bets = df[df["is_value"]]
    value_profit = value_bets.apply(
        lambda r: (r["sp"] - 1) if r["won"] else -1.0, axis=1
    )
    value_pnl = float(value_profit.sum()) if len(value_profit) > 0 else 0
    value_winners = int(value_bets["won"].sum())

    # --- Strategy 3: Kelly criterion staking (fractional Kelly) ---
    # Use a hybrid approach: bet on the model's top pick per race,
    # but size the bet using Kelly based on the probability edge.
    # We use a lower min-edge threshold than value betting since we're
    # already filtering to the model's top pick (higher conviction).
    kelly_fraction = 0.25  # Quarter Kelly for safety
    kelly_min_edge = 0.02  # 2% min edge (lower than value betting's 5%)
    bankroll = 100.0
    kelly_results = []
    for race_id, group in df.groupby("race_id"):
        if len(group) == 0:
            continue
        top = group.loc[group["prob"].idxmax()]
        b = top["sp"] - 1
        if b <= 0:
            continue

        # Check if there's a minimum probability edge
        edge = top["prob"] - top["implied_prob"]
        if edge < kelly_min_edge:
            continue

        f_star = (b * top["prob"] - (1 - top["prob"])) / b
        if f_star <= 0:
            continue
        stake_pct = min(f_star * kelly_fraction, 0.05)  # Cap at 5% of bankroll
        stake = bankroll * stake_pct
        profit = stake * (top["sp"] - 1) if top["won"] else -stake
        kelly_results.append({
            "race_id": int(race_id),
            "won": bool(top["won"]),
            "stake": round(float(stake), 2),
            "profit": round(float(profit), 2),
            "edge": round(float(edge), 4),
        })

    kelly_total_staked = sum(r["stake"] for r in kelly_results) if kelly_results else 0
    kelly_pnl = sum(r["profit"] for r in kelly_results) if kelly_results else 0

    fav_pnl = sum(fav_profits) if fav_profits else 0

    # Cumulative P&L for charting (includes favourite baseline)
    cumulative = []
    running = 0.0
    fav_running = 0.0
    for i, r in enumerate(top_pick_results):
        running += r["profit"]
        fav_running += fav_profits[i]
        cumulative.append({
            "race": len(cumulative) + 1,
            "pnl": round(running, 2),
            "fav_pnl": round(fav_running, 2),
        })

    # Kelly cumulative P&L for charting
    kelly_cumulative = []
    kelly_running = 0.0
    for r in kelly_results:
        kelly_running += r["profit"]
        kelly_cumulative.append({"race": len(kelly_cumulative) + 1, "pnl": round(kelly_running, 2)})

    return {
        "top_pick_pnl": round(top_pick_pnl, 2),
        "top_pick_roi": round(top_pick_pnl / max(top_pick_races, 1) * 100, 2),
        "top_pick_races": top_pick_races,
        "top_pick_winners": top_pick_winners,
        "top_pick_strike_rate": round(top_pick_winners / max(top_pick_races, 1) * 100, 1),
        "value_bet_pnl": round(value_pnl, 2),
        "value_bet_roi": round(value_pnl / max(len(value_bets), 1) * 100, 2),
        "value_bet_count": len(value_bets),
        "value_bet_winners": value_winners,
        "kelly_pnl": round(kelly_pnl, 2),
        "kelly_roi": round(kelly_pnl / max(kelly_total_staked, 1) * 100, 2),
        "kelly_races": len(kelly_results),
        "kelly_total_staked": round(kelly_total_staked, 2),
        "kelly_pnl_by_race": kelly_cumulative,
        "favourite_pnl": round(fav_pnl, 2),
        "favourite_roi": round(fav_pnl / max(len(fav_profits), 1) * 100, 2),
        "pnl_by_race": cumulative,
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
