"""Bankroll management models: track bets, outcomes, and bankroll history."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, ForeignKey
from app.database import Base


class BankrollConfig(Base):
    """User's bankroll configuration (singleton-ish — one active config)."""
    __tablename__ = "bankroll_config"

    id = Column(Integer, primary_key=True, index=True)
    initial_bankroll = Column(Float, nullable=False, default=100.0)
    current_bankroll = Column(Float, nullable=False, default=100.0)
    kelly_fraction = Column(Float, nullable=False, default=0.25)  # quarter Kelly
    min_edge = Column(Float, nullable=False, default=0.05)  # 5% minimum edge
    max_stake_pct = Column(Float, nullable=False, default=0.05)  # 5% max of bankroll
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BetRecord(Base):
    """Record of a bet placed (or simulated)."""
    __tablename__ = "bet_records"

    id = Column(Integer, primary_key=True, index=True)
    race_entry_id = Column(Integer, ForeignKey("race_entries.id"), nullable=False, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)
    dog_name = Column(String)
    track_name = Column(String)
    race_date = Column(String)
    race_number = Column(Integer)
    trap = Column(Integer)
    grade = Column(String)

    # Bet type discriminator.  "win" matches every legacy row (the
    # alembic migration backfills NULL -> "win"). Combo bets use
    # "place", "show", "forecast", "trio" and stash their constituent
    # entry ids in `legs_json`.
    bet_type = Column(String, nullable=True, default="win")
    legs_json = Column(Text, nullable=True)

    # Bet details
    win_probability = Column(Float)
    # For combo bets this is the combo probability (forecast/trio P).
    combo_probability = Column(Float, nullable=True)
    odds_decimal = Column(Float)
    implied_prob = Column(Float)
    edge = Column(Float)
    confidence_tier = Column(String)

    # Staking
    stake = Column(Float, nullable=False)
    stake_method = Column(String, default="kelly")  # "kelly", "flat", "manual"
    bankroll_before = Column(Float)

    # Outcome (null until race results)
    outcome = Column(String)  # "pending", "won", "lost"
    profit = Column(Float)  # (odds - 1) * stake if won, else -stake
    bankroll_after = Column(Float)

    settled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
