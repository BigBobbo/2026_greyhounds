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
        # The headline "accuracy" number is misleading on its own: with a
        # ~17% winner base rate, predicting "nobody wins" scores ~83%.
        # Surface the base rate so accuracy is always read against it.
        metrics["base_rate"] = float(np.mean(y_true))

        if y_proba is not None:
            # Calibration slope/intercept: logistic refit of outcomes on the
            # model's own log-odds. Perfect calibration = slope 1, intercept
            # 0; slope < 1 means over-confident probabilities.
            try:
                from sklearn.linear_model import LogisticRegression

                p = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
                logit = np.log(p / (1 - p)).reshape(-1, 1)
                if len(np.unique(y_true)) == 2:
                    lr = LogisticRegression(C=1e6, solver="lbfgs")
                    lr.fit(logit, np.asarray(y_true).astype(int))
                    metrics["calibration_slope"] = float(lr.coef_[0][0])
                    metrics["calibration_intercept"] = float(lr.intercept_[0])
            except Exception:
                pass

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
    commission_rate: float = 0.05,
    slippage: float = 0.05,
    min_odds: float = 1.5,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """
    Compute betting P&L metrics at REALISTIC execution prices.

    The old simulation bet at the final post-race SP with no commission —
    a price nobody can obtain in advance. Here every bet executes at
    ``price_taken = 1 + (SP - 1) * (1 - slippage)`` with ``commission_rate``
    charged on net winnings, so backtested edges are systematically
    conservative rather than systematically flattering. Value/Kelly
    strategies also refuse prices below ``min_odds``.

    Race-level bootstrap confidence intervals (90%) are reported for every
    ROI so a "profitable" strategy whose CI straddles zero is visibly
    noise, not signal.

    Simulates: $1 on the model's top pick per race; $1 value bets (model
    prob exceeds implied by min edge); Kelly staking via the canonical
    ml.staking module; and a $1 SP-favourite baseline under the same
    execution model.
    """
    import pandas as pd

    def _take_price(sp: float) -> float:
        return 1.0 + (sp - 1.0) * (1.0 - slippage)

    def _profit(sp: float, won: bool, stake: float = 1.0) -> float:
        if not won:
            return -stake
        return stake * (_take_price(sp) - 1.0) * (1.0 - commission_rate)

    def _roi_ci(profits: list[float], stakes: list[float]) -> list[float] | None:
        """90% bootstrap CI on ROI%, resampling races with replacement."""
        n = len(profits)
        if n < 20:
            return None
        rng = np.random.default_rng(42)
        p = np.asarray(profits)
        s = np.asarray(stakes)
        rois = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            staked = s[idx].sum()
            if staked > 0:
                rois.append(p[idx].sum() / staked * 100.0)
        if not rois:
            return None
        return [
            round(float(np.percentile(rois, 5)), 2),
            round(float(np.percentile(rois, 95)), 2),
        ]

    df = pd.DataFrame({
        "won": y_true.astype(bool),
        "prob": y_proba,
        "sp": sp_decimal,
        "race_id": race_ids,
    })

    # Normalize probabilities within each race BEFORE any filtering, so the
    # backtest bets on the same scale the serving path produces (exactly one
    # dog wins a race; per-race probabilities must sum to 1). Without this
    # the backtested edge is systematically different from the served edge.
    sums = df.groupby("race_id")["prob"].transform("sum")
    df.loc[sums > 0, "prob"] = df.loc[sums > 0, "prob"] / sums[sums > 0]

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
        profit = _profit(float(top["sp"]), bool(top["won"]))
        top_pick_results.append({
            "race_id": int(race_id),
            "won": bool(top["won"]),
            "sp": float(top["sp"]),
            "prob": float(top["prob"]),
            "profit": float(profit),
        })
        # Favourite = lowest SP (baseline) — same execution model
        fav = group.loc[group["sp"].idxmin()]
        fav_profits.append(_profit(float(fav["sp"]), bool(fav["won"])))

    top_pick_df = pd.DataFrame(top_pick_results)
    top_pick_pnl = float(top_pick_df["profit"].sum()) if len(top_pick_df) > 0 else 0
    top_pick_races = len(top_pick_df)
    top_pick_winners = int(top_pick_df["won"].sum()) if len(top_pick_df) > 0 else 0

    # --- Strategy 2: Value betting ($1 when model prob clears implied + edge) ---
    min_edge = 0.05
    df["edge"] = df["prob"] - df["implied_prob"]
    df["is_value"] = (df["edge"] > min_edge) & (df["sp"] >= min_odds)
    value_bets = df[df["is_value"]]
    value_profit = value_bets.apply(
        lambda r: _profit(float(r["sp"]), bool(r["won"])), axis=1
    )
    value_pnl = float(value_profit.sum()) if len(value_profit) > 0 else 0
    value_winners = int(value_bets["won"].sum())

    # --- Strategy 3: Kelly staking via the canonical staking module ---
    # Bet the model's top pick per race, sized by the same code that sizes
    # real recommendations at serve time — commission, min-odds and caps
    # included — so the backtested strategy IS the served strategy.
    from ml.staking import StakingConfig, kelly_stake

    kelly_cfg = StakingConfig(
        bankroll=100.0,
        kelly_fraction=0.25,
        min_edge=0.02,  # top-pick pre-filter carries conviction; lower floor
        max_stake_pct=0.05,
        commission_rate=commission_rate,
        min_odds=min_odds,
    )
    kelly_results = []
    for race_id, group in df.groupby("race_id"):
        if len(group) == 0:
            continue
        top = group.loc[group["prob"].idxmax()]
        # Size against the obtainable price, not the closing SP
        rec = kelly_stake(float(top["prob"]), _take_price(float(top["sp"])), kelly_cfg)
        if not rec.get("bet"):
            continue
        stake = rec["stake"]
        # Winnings already priced at the taken price; commission applied here
        profit = (
            stake * (_take_price(float(top["sp"])) - 1.0) * (1.0 - commission_rate)
            if top["won"] else -stake
        )
        kelly_results.append({
            "race_id": int(race_id),
            "won": bool(top["won"]),
            "stake": round(float(stake), 2),
            "profit": round(float(profit), 2),
            "edge": rec.get("edge"),
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
        "top_pick_roi_ci90": _roi_ci(
            [r["profit"] for r in top_pick_results],
            [1.0] * len(top_pick_results),
        ),
        "top_pick_races": top_pick_races,
        "top_pick_winners": top_pick_winners,
        "top_pick_strike_rate": round(top_pick_winners / max(top_pick_races, 1) * 100, 1),
        "value_bet_pnl": round(value_pnl, 2),
        "value_bet_roi": round(value_pnl / max(len(value_bets), 1) * 100, 2),
        "value_bet_roi_ci90": _roi_ci(
            list(value_profit) if len(value_bets) else [],
            [1.0] * len(value_bets),
        ),
        "value_bet_count": len(value_bets),
        "value_bet_winners": value_winners,
        "kelly_pnl": round(kelly_pnl, 2),
        "kelly_roi": round(kelly_pnl / max(kelly_total_staked, 1) * 100, 2),
        "kelly_roi_ci90": _roi_ci(
            [r["profit"] for r in kelly_results],
            [r["stake"] for r in kelly_results],
        ),
        "kelly_races": len(kelly_results),
        "kelly_total_staked": round(kelly_total_staked, 2),
        "kelly_pnl_by_race": kelly_cumulative,
        "favourite_pnl": round(fav_pnl, 2),
        "favourite_roi": round(fav_pnl / max(len(fav_profits), 1) * 100, 2),
        "favourite_roi_ci90": _roi_ci(fav_profits, [1.0] * len(fav_profits)),
        "pnl_by_race": cumulative,
        "execution_model": {
            "commission_rate": commission_rate,
            "slippage": slippage,
            "min_odds": min_odds,
            "note": (
                "All P&L at price 1+(SP-1)*(1-slippage) with commission on "
                "net winnings — SP itself is unobtainable in advance."
            ),
        },
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
