from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, JSON
from app.database import Base


class FeatureDefinition(Base):
    __tablename__ = "feature_definitions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    description = Column(Text)
    feature_type = Column(String, nullable=False)  # "visual" or "code"
    config_json = Column(JSON)  # for visual features
    code = Column(Text)  # for code features
    input_columns = Column(JSON)  # ["finish_position", "finish_time", ...]
    output_dtype = Column(String, default="float")
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
