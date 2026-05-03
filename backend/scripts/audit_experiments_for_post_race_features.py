"""Audit which existing experiments use post-race-only features.

Loads every completed experiment's saved model artifact, intersects its
trained `feature_names` with `ml.feature_availability.POST_RACE_FEATURE_NAMES`,
and prints a per-experiment report. Models that show any matches will fail
the new strict-mode guard in `predict_race` when used on a scheduled race
and need to be retrained with `exclude_post_race_features=True` (the
default for new experiments).

Run from the backend directory:
    python scripts/audit_experiments_for_post_race_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.models.experiment import Experiment
from app.services.prediction_service import load_trained_model
from ml.feature_availability import POST_RACE_FEATURE_NAMES


def main() -> int:
    db = SessionLocal()
    try:
        experiments = (
            db.query(Experiment)
            .filter(Experiment.status == "completed")
            .order_by(Experiment.id.desc())
            .all()
        )
        if not experiments:
            print("No completed experiments found.")
            return 0

        affected = 0
        for exp in experiments:
            try:
                artifact = load_trained_model(exp)
            except FileNotFoundError:
                print(f"#{exp.id} {exp.name!r}: model artifact missing, skipped")
                continue
            except Exception as e:
                print(f"#{exp.id} {exp.name!r}: failed to load ({e}), skipped")
                continue

            feature_names = artifact.get("feature_names") or []
            offenders = sorted(set(feature_names) & set(POST_RACE_FEATURE_NAMES))
            status = "BLOCKED" if offenders else "ok"
            print(
                f"#{exp.id:<4} {exp.name!r:<40} "
                f"target={exp.target:<14} status={status}"
            )
            if offenders:
                affected += 1
                for name in offenders:
                    print(f"        - {name}: {POST_RACE_FEATURE_NAMES[name]}")

        print()
        print(
            f"Summary: {affected} of {len(experiments)} completed experiments "
            f"will refuse to serve scheduled races under the new strict guard."
        )
        if affected:
            print(
                "Retrain those with exclude_post_race_features=True (the "
                "default in dataset_builder.build_dataset) to clear the "
                "block."
            )
        return 1 if affected else 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
