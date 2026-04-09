from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, JSON
from app.database import Base


class FeatureVersion(Base):
    __tablename__ = "feature_versions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Snapshot of data-integrity report at time of materialization
    coverage_snapshot = Column(JSON)
