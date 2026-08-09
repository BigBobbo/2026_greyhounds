"""Register the committed retrain model as an experiment — idempotent.

A fresh container rebuilds the local DB from data_mirror dumps, which
don't carry the locally-registered experiment row. This recreates it so
prediction scripts can reference the model by name. Prints the experiment
id either way.

Usage (from backend/):
    DATABASE_URL=... python3 scripts/register_retrain_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal  # noqa: E402
from app.models.experiment import Experiment  # noqa: E402

NAME = "retrain-2026-08-01"
# models_store/ is inside the git tree (and the Docker image) — data/ is
# not: it's gitignored locally and shadowed by the volume mount in prod.
MODEL_PATH = os.path.join("models_store", "retrain_model.joblib")


def main() -> int:
    db = SessionLocal()
    try:
        existing = db.query(Experiment).filter(Experiment.name == NAME).first()
        if existing:
            print(f"already registered: experiment id={existing.id}")
            return existing.id
        if not os.path.exists(MODEL_PATH):
            raise SystemExit(f"model artifact missing: {MODEL_PATH}")

        import joblib
        bundle = joblib.load(MODEL_PATH)
        exp = Experiment(
            name=NAME,
            description=(
                "Definitive honest retrain on enriched mirror (sectionals, "
                "pedigree, trainer form, weather). Blend alpha/beta in artifact"
            ),
            algorithm="lightgbm",
            target="win_prob",
            hyperparameters={"n_estimators": 600, "learning_rate": 0.05,
                             "num_leaves": 63},
            feature_set=[],
            split_config={"max_entries": 250000, "val_pct": 0.12,
                          "test_pct": 0.12},
            status="completed",
            metrics={"blend_alpha": bundle.get("blend_alpha"),
                     "blend_beta": bundle.get("blend_beta")},
            model_path=MODEL_PATH,
        )
        db.add(exp)
        db.commit()
        print(f"registered: experiment id={exp.id}")
        return exp.id
    finally:
        db.close()


if __name__ == "__main__":
    main()
