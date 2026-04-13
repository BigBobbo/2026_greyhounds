from sqlalchemy import Boolean, Column, Float, Integer, String, ForeignKey, UniqueConstraint
from app.database import Base


class RaceEntry(Base):
    __tablename__ = "race_entries"

    id = Column(Integer, primary_key=True, index=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False, index=True)
    dog_id = Column(Integer, ForeignKey("dogs.id"), nullable=False, index=True)
    trap = Column(Integer, nullable=False)
    finish_position = Column(Integer, index=True)
    finish_time = Column(Float)  # seconds
    sectional_time = Column(Float)  # time to first bend
    adjusted_time = Column(Float)  # after going allowance
    beaten_distance = Column(Float)  # lengths behind winner
    weight_kg = Column(Float)
    starting_price = Column(String)  # "3/1", "evens"
    sp_decimal = Column(Float)  # 4.0, 2.0
    comment = Column(String)
    wide_runner = Column(Boolean, default=False)
    grade_at_entry = Column(String)
    days_since_last = Column(Integer)

    __table_args__ = (
        UniqueConstraint("race_id", "trap", name="uq_entry_race_trap"),
    )
