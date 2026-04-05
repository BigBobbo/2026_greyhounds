from datetime import datetime
from typing import Any
from pydantic import BaseModel


class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None
    algorithm: str  # "xgboost", "lightgbm", "logistic_regression", "random_forest"
    target: str  # "win_prob", "finish_position", "finish_time"
    hyperparameters: dict[str, Any]
    feature_set: list[int]  # list of feature_definition IDs
    split_config: dict[str, Any] | None = None
    auto_tune: bool = False
    optuna_trials: int = 50


class ExperimentResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    algorithm: str
    target: str
    hyperparameters: dict[str, Any]
    feature_set: list[int]
    split_config: dict[str, Any] | None = None
    status: str
    metrics: dict[str, Any] | None = None
    confusion_matrix: Any | None = None
    calibration_data: Any | None = None
    roc_data: Any | None = None
    shap_summary: Any | None = None
    feature_importance: Any | None = None
    training_duration_s: float | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    id: int
    experiment_id: int
    race_entry_id: int
    win_probability: float | None = None
    predicted_position: float | None = None
    predicted_time: float | None = None
    confidence: float | None = None
    dog_name: str | None = None
    trap: int | None = None

    model_config = {"from_attributes": True}
