from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, ForeignKey, UniqueConstraint, JSON
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)
    race_entry_id = Column(Integer, ForeignKey("race_entries.id"), nullable=False, index=True)
    win_probability = Column(Float)
    predicted_position = Column(Float)
    predicted_time = Column(Float)
    confidence = Column(Float)

    # Place/show marginals from the Plackett-Luce / Henery position
    # distribution.  Null for trainers that don't emit a full position
    # breakdown (XGBoost, LightGBM pointwise, sklearn).
    place2_probability = Column(Float)
    place3_probability = Column(Float)
    # {"p1": 0.34, "p2": 0.21, "p3": 0.15, "p4_plus": 0.30}
    position_probs_json = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("experiment_id", "race_entry_id", name="uq_prediction_exp_entry"),
    )
