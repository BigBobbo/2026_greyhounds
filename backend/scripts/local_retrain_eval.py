"""Honest retrain + evaluation on the local production mirror.

The first evaluation this project has ever run with:
  * point-in-time aggregates (no future data in any feature),
  * a chronological train / val / test split with all of a race together,
  * per-race probability normalization identical to serving,
  * betting P&L at realistic execution prices (slippage + commission,
    min-odds floor) with bootstrap confidence intervals,
  * the SP-favourite baseline under the same execution model.

Usage (from backend/):
    DATABASE_URL=sqlite:///./data/greyhound_local.db \
        python3 scripts/local_retrain_eval.py [--max-entries 250000]

Writes a JSON report to data/retrain_report.json and prints a summary.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.database import SessionLocal  # noqa: E402


def main(max_entries: int, test_pct: float, val_pct: float) -> None:
    from ml.dataset_builder import build_dataset
    from ml.evaluation import compute_betting_metrics, compute_metrics
    from ml.trainers.lightgbm_trainer import LightGBMTrainer

    db = SessionLocal()
    t0 = time.time()
    print(f"Building dataset (max_entries={max_entries})...", flush=True)
    data = build_dataset(
        db,
        feature_ids=[],
        target="win_prob",
        split_config={
            "max_entries": max_entries,
            "val_pct": val_pct,
            "test_pct": test_pct,
        },
        include_builtin_features=True,
        include_elo_features=True,
        include_h2h_features=True,
        include_sp_features=False,   # post-race — the guard would drop them
        include_race_relative_features=True,
        include_pace_shape_features=True,
        impute_missing=False,        # LightGBM handles NaN natively
        exclude_post_race_features=True,
    )
    print(
        f"Dataset built in {time.time()-t0:.0f}s: "
        f"train={len(data['X_train'])} val={len(data['X_val'])} "
        f"test={len(data['X_test'])} features={len(data['feature_names'])}",
        flush=True,
    )

    trainer = LightGBMTrainer({
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    })
    t1 = time.time()
    result = trainer.train(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"],
    )
    print(f"Trained in {time.time()-t1:.0f}s: {result.metrics}", flush=True)

    # --- Honest test evaluation ---
    X_test, y_test = data["X_test"], data["y_test"]
    meta_test = data["meta_test"]
    proba = trainer.predict_proba(X_test, calibrate=True)

    y_arr = np.asarray(y_test).astype(int)
    clf_metrics = compute_metrics(
        y_arr, (np.asarray(proba) > 0.5).astype(int), np.asarray(proba),
        "classification",
    )
    betting = compute_betting_metrics(
        y_arr,
        np.asarray(proba),
        meta_test["sp_decimal"].values,
        meta_test["race_id"].values,
    )
    # Drop the bulky per-race series from the printed/stored report
    betting.pop("pnl_by_race", None)
    betting.pop("kelly_pnl_by_race", None)

    # --- Benter blend: second-stage model+market conditional logit ---
    # Fit alpha/beta on the VALIDATION window, apply to test. Market
    # probabilities come from de-vigged SPs (the same price source the
    # betting sim executes against, minus slippage).
    from ml.blend import BlendModel, devig_market_probs, fit_blend

    meta_val = data["meta_val"]
    val_proba = trainer.predict_proba(data["X_val"], calibrate=True)
    val_market = devig_market_probs(
        meta_val["sp_decimal"].values, meta_val["race_id"].values,
    )
    blender = fit_blend(
        np.asarray(val_proba), val_market,
        np.asarray(data["y_val"]).astype(int), meta_val["race_id"].values,
    )
    test_market = devig_market_probs(
        meta_test["sp_decimal"].values, meta_test["race_id"].values,
    )
    blended = blender.blend(
        np.asarray(proba), test_market, meta_test["race_id"].values,
    )
    betting_blended = compute_betting_metrics(
        y_arr, blended,
        meta_test["sp_decimal"].values,
        meta_test["race_id"].values,
    )
    betting_blended.pop("pnl_by_race", None)
    betting_blended.pop("kelly_pnl_by_race", None)
    blend_clf = compute_metrics(
        y_arr, (blended > 0.5).astype(int), blended, "classification",
    )

    # Persist the trained artifacts for downstream use (bet sheets, blend)
    import joblib

    joblib.dump(
        {
            "trainer": trainer,
            "feature_names": data["feature_names"],
            "feature_medians": data["feature_medians"],
            "nan_policy": "passthrough",
            "is_ranking": False,
            "blend_alpha": blender.alpha,
            "blend_beta": blender.beta,
        },
        os.path.join("data", "retrain_model.joblib"),
    )

    report = {
        "dataset": data["stats"],
        "n_features": len(data["feature_names"]),
        "train_metrics": result.metrics,
        "test_classification": clf_metrics,
        "test_betting": betting,
        "blend": {"alpha": blender.alpha, "beta": blender.beta},
        "test_classification_blended": blend_clf,
        "test_betting_blended": betting_blended,
        "test_period": {
            "from": str(meta_test["race_date"].min()),
            "to": str(meta_test["race_date"].max()),
            "races": int(meta_test["race_id"].nunique()),
        },
        "top_features": dict(sorted(
            trainer.get_feature_importance().items(),
            key=lambda kv: kv[1], reverse=True,
        )[:30]) if hasattr(trainer, "get_feature_importance") else {},
    }

    out = os.path.join("data", "retrain_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1, default=str)

    print(json.dumps({k: v for k, v in report.items() if k != "top_features"},
                     indent=1, default=str), flush=True)
    print(f"\nReport written to {out}", flush=True)
    db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entries", type=int, default=250000)
    ap.add_argument("--test-pct", type=float, default=0.12)
    ap.add_argument("--val-pct", type=float, default=0.12)
    args = ap.parse_args()
    main(args.max_entries, args.test_pct, args.val_pct)
