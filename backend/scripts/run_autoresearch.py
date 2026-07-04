#!/usr/bin/env python3
"""
CLI entry point for the autoresearch loop.

Usage:
    # Run 100 experiments optimizing Kelly ROI with all enabled features
    python scripts/run_autoresearch.py

    # Custom objective and limits
    python scripts/run_autoresearch.py --objective betting_top_pick_roi --max 50 --patience 15

    # Specific features and algorithm
    python scripts/run_autoresearch.py --features 1,2,3,5,8 --algorithm xgboost

    # List available objectives
    python scripts/run_autoresearch.py --list-objectives
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models.feature_definition import FeatureDefinition
from ml.autoresearch import OBJECTIVE_DIRECTIONS, AutoResearchLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("autoresearch")


def get_enabled_feature_ids(db) -> list[int]:
    """Get all enabled feature definition IDs."""
    features = (
        db.query(FeatureDefinition)
        .filter(FeatureDefinition.enabled.is_(True))
        .all()
    )
    return [f.id for f in features]


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch: autonomous ML experiment loop for greyhound prediction",
    )
    parser.add_argument(
        "--objective",
        default="betting_kelly_roi",
        help=f"Metric to optimize (default: betting_kelly_roi). Options: {', '.join(OBJECTIVE_DIRECTIONS)}",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=100,
        dest="max_experiments",
        help="Maximum experiments to run (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Stop after N experiments without improvement (default: 20)",
    )
    parser.add_argument(
        "--algorithm",
        default="lightgbm",
        choices=["xgboost", "lightgbm", "lambdarank", "random_forest"],
        help="Starting algorithm (default: lightgbm)",
    )
    parser.add_argument(
        "--target",
        default="win_prob",
        choices=["win_prob", "finish_position", "finish_time"],
        help="Prediction target (default: win_prob)",
    )
    parser.add_argument(
        "--features",
        help="Comma-separated feature IDs (default: all enabled features)",
    )
    parser.add_argument(
        "--list-objectives",
        action="store_true",
        help="List available objectives and exit",
    )
    parser.add_argument(
        "--output",
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    if args.list_objectives:
        print("Available objectives:")
        for obj, higher in OBJECTIVE_DIRECTIONS.items():
            direction = "maximize" if higher else "minimize"
            print(f"  {obj:40s} ({direction})")
        return

    db = SessionLocal()
    try:
        # Resolve feature IDs
        if args.features:
            feature_ids = [int(x.strip()) for x in args.features.split(",")]
        else:
            feature_ids = get_enabled_feature_ids(db)

        if not feature_ids:
            logger.error("No features found. Seed features first: python scripts/seed_features.py")
            sys.exit(1)

        all_feature_ids = get_enabled_feature_ids(db)

        logger.info("Starting autoresearch loop")
        logger.info("  Objective:    %s", args.objective)
        logger.info("  Algorithm:    %s", args.algorithm)
        logger.info("  Target:       %s", args.target)
        logger.info("  Features:     %d starting, %d searchable", len(feature_ids), len(all_feature_ids))
        logger.info("  Max trials:   %d", args.max_experiments)
        logger.info("  Patience:     %d", args.patience)

        def on_improvement(trial, score, proposal):
            logger.info(
                "  >> New best at trial %d: %s=%.4f (strategy=%s, algo=%s)",
                trial, args.objective, score,
                proposal["strategy"], proposal["algorithm"],
            )

        loop = AutoResearchLoop(
            db=db,
            feature_ids=feature_ids,
            objective=args.objective,
            algorithm=args.algorithm,
            target=args.target,
            all_feature_ids=all_feature_ids,
        )

        summary = loop.run(
            max_experiments=args.max_experiments,
            patience=args.patience,
            on_improvement=on_improvement,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("AUTORESEARCH RESULTS")
        print("=" * 60)
        print(f"  Total experiments:   {summary['total_experiments']}")
        print(f"  Improvements found:  {summary['total_improvements']}")
        print(f"  Improvement rate:    {summary['improvement_rate']}%")
        print(f"  Total duration:      {summary['total_duration_s']:.0f}s")
        print(f"  Best {args.objective}: {summary['best_score']:.4f}")
        print(f"  Best algorithm:      {summary['best_algorithm']}")
        print(f"  Best features:       {summary['best_feature_set']}")
        print("  Best hyperparams:")
        for k, v in summary["best_hyperparameters"].items():
            print(f"    {k}: {v}")
        print()
        print("Strategy breakdown:")
        for strategy, count in summary["strategy_counts"].items():
            wins = summary["strategy_improvements"].get(strategy, 0)
            print(f"  {strategy:30s}  {wins}/{count} improvements")
        print("=" * 60)

        # Save results
        if args.output:
            output_path = Path(args.output)
            # Strip non-serializable history details for JSON
            save_summary = {k: v for k, v in summary.items() if k != "history"}
            output_path.write_text(json.dumps(save_summary, indent=2, default=str))
            logger.info("Results saved to %s", output_path)

    finally:
        db.close()


if __name__ == "__main__":
    main()
