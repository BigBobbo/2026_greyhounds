from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, JSON
from app.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    algorithm = Column(String, nullable=False)  # "xgboost", "lightgbm", etc.
    target = Column(String, nullable=False)  # "win_prob", "finish_position", "finish_time"
    hyperparameters = Column(JSON, nullable=False)
    feature_set = Column(JSON, nullable=False)  # list of feature_definition IDs
    split_config = Column(JSON)
    status = Column(String, default="pending")  # pending / running / completed / failed
    metrics = Column(JSON)
    confusion_matrix = Column(JSON)
    calibration_data = Column(JSON)
    roc_data = Column(JSON)
    shap_summary = Column(JSON)
    feature_importance = Column(JSON)
    training_duration_s = Column(Float)
    model_path = Column(String)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
