from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, ForeignKey
from app.database import Base


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False, index=True)
    dog_id = Column(Integer, ForeignKey("dogs.id"), nullable=False, index=True)
    bookmaker = Column(String, nullable=False)
    odds_decimal = Column(Float, nullable=False)
    implied_prob = Column(Float)
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_sp = Column(Boolean, default=False)

    # Betfair identifiers for the market/runner this price came from.
    # Kept so the same market can be re-listed after the off to settle its
    # Betfair Starting Price — a closed market cannot be found again from
    # (race, dog) alone.
    market_id = Column(String, index=True)
    selection_id = Column(Integer)
