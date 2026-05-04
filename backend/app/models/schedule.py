from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)

from app.database import Base


class ModelSchedule(Base):
    """One row per experiment the user wants to run on a daily schedule."""

    __tablename__ = "model_schedule"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(
        Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    enabled = Column(Boolean, nullable=False, default=True)
    is_main = Column(Boolean, nullable=False, default=False)
    cron_hour = Column(Integer, nullable=False, default=8)
    cron_minute = Column(Integer, nullable=False, default=30)
    timezone = Column(String, nullable=False, default="Europe/Dublin")
    scrape_upcoming = Column(Boolean, nullable=False, default=True)
    predict_days_ahead = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ScheduledPredictionRun(Base):
    """Append-only audit log of one execution of a scheduled prediction job."""

    __tablename__ = "scheduled_prediction_run"

    id = Column(Integer, primary_key=True, index=True)
    model_schedule_id = Column(
        Integer,
        ForeignKey("model_schedule.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run_date = Column(Date, nullable=False, index=True)
    status = Column(String, nullable=False)  # running | success | partial | failed
    trigger = Column(String, nullable=False, default="scheduled")  # scheduled | manual
    races_predicted = Column(Integer, nullable=False, default=0)
    races_skipped = Column(Integer, nullable=False, default=0)
    predictions_written = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
