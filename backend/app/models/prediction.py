from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
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

    # Multi-position probabilities derived from the ordering service.
    # place = P(finish 1st or 2nd), show = P(finish 1st, 2nd, or 3rd).
    place_probability = Column(Float, nullable=True)
    show_probability = Column(Float, nullable=True)

    # JSON-encoded cached top forecast/trio combos for this race. Stored
    # on every prediction row in the race so a single-row API fetch can
    # render the combos panel without a separate per-race query.
    forecast_combos_json = Column(Text, nullable=True)
    trio_combos_json = Column(Text, nullable=True)

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
