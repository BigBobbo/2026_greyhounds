from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Time, ForeignKey, UniqueConstraint
from app.database import Base


class Race(Base):
    __tablename__ = "races"

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    race_date = Column(Date, nullable=False, index=True)
    race_time = Column(Time)
    race_number = Column(Integer)
    distance_m = Column(Integer, nullable=False)
    grade = Column(String)  # "A1", "A2", "S1", "OR", etc.
    race_type = Column(String, default="flat")  # "flat", "hurdle"
    prize_money = Column(Float)
    going = Column(String)  # "standard", "slow", "fast"
    going_allowance = Column(Float)  # seconds adjustment
    num_runners = Column(Integer)
    source = Column(String)  # "gri", "greyhound_data", "timeform"
    source_id = Column(String)
    status = Column(String, default="scheduled", index=True)  # scheduled / resulted / void
    created_at = Column(DateTime, default=datetime.utcnow)
    last_scraped_at = Column(DateTime)
    last_scrape_log_id = Column(Integer, ForeignKey("scrape_logs.id"), index=True)

    __table_args__ = (
        UniqueConstraint("track_id", "race_date", "race_number", name="uq_race_track_date_num"),
    )
