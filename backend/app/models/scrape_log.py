from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from app.database import Base


class ScrapeLog(Base):
    __tablename__ = "scrape_logs"

    id = Column(Integer, primary_key=True, index=True)
    spider_name = Column(String, nullable=False)  # "gri", "greyhound_data", "odds"
    source = Column(String)  # URL or description
    status = Column(String, default="running")  # running / success / failed / partial
    records_scraped = Column(Integer, default=0)
    records_new = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    heartbeat_at = Column(DateTime)
    completed_at = Column(DateTime)
