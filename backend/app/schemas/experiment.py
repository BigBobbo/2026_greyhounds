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


class ExperimentListItem(BaseModel):
    """Lightweight schema for list endpoints — excludes heavy fields like training_log."""
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
    feature_importance: Any | None = None
    training_duration_s: float | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    training_stage: str | None = None

    model_config = {"from_attributes": True}


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
    training_log: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    training_stage: str | None = None

    model_config = {"from_attributes": True}


class PredictionResponse(BaseModel):
    id: int
    experiment_id: int
    race_entry_id: int
    win_probability: float | None = None
    predicted_position: float | None = None
    predicted_time: float | None = None
    confidence: float | None = None
    confidence_tier: str | None = None
    margin: float | None = None
    entropy: float | None = None
    edge: float | None = None
    is_value: bool | None = None
    kelly_bet: bool | None = None
    kelly_stake: float | None = None
    kelly_stake_pct: float | None = None
    kelly_full_pct: float | None = None
    kelly_expected_value: float | None = None
    kelly_implied_prob: float | None = None
    kelly_reason: str | None = None
    data_completeness: float | None = None
    bankroll_used: float | None = None
    sp_decimal_at_pred: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    dog_name: str | None = None
    trap: int | None = None

    model_config = {"from_attributes": True}
