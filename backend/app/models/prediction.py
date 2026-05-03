from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)
    race_entry_id = Column(Integer, ForeignKey("race_entries.id"), nullable=False, index=True)

    # Core scores
    win_probability = Column(Float)
    predicted_position = Column(Float)
    predicted_time = Column(Float)
    confidence = Column(Float)

    # Race-level confidence breakdown (replicates what predict_race returns
    # so saved predictions can be replayed without recomputing the model).
    confidence_tier = Column(String, nullable=True)
    margin = Column(Float, nullable=True)
    entropy = Column(Float, nullable=True)

    # Market comparison
    edge = Column(Float, nullable=True)
    is_value = Column(Boolean, nullable=True)

    # Kelly staking snapshot — captured against the bankroll the user chose
    # at prediction time so historical recommendations stay reproducible.
    kelly_bet = Column(Boolean, nullable=True)
    kelly_stake = Column(Float, nullable=True)
    kelly_stake_pct = Column(Float, nullable=True)
    kelly_full_pct = Column(Float, nullable=True)
    kelly_expected_value = Column(Float, nullable=True)
    kelly_implied_prob = Column(Float, nullable=True)
    kelly_reason = Column(String, nullable=True)
    sp_decimal_at_pred = Column(Float, nullable=True)

    # Reproduction context
    data_completeness = Column(Float, nullable=True)
    bankroll_used = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("experiment_id", "race_entry_id", name="uq_prediction_exp_entry"),
    )
