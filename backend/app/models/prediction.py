from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, ForeignKey, UniqueConstraint
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
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("experiment_id", "race_entry_id", name="uq_prediction_exp_entry"),
    )
