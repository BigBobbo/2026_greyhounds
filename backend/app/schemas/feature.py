from datetime import datetime
from typing import Any
from pydantic import BaseModel


class FeatureDefinitionBase(BaseModel):
    name: str
    display_name: str | None = None
    description: str | None = None
    feature_type: str  # "visual" or "code"
    config_json: dict[str, Any] | None = None
    code: str | None = None
    input_columns: list[str] | None = None
    output_dtype: str = "float"
    enabled: bool = True


class FeatureDefinitionCreate(FeatureDefinitionBase):
    pass


class FeatureDefinitionUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    config_json: dict[str, Any] | None = None
    code: str | None = None
    input_columns: list[str] | None = None
    enabled: bool | None = None


class FeatureDefinitionResponse(FeatureDefinitionBase):
    id: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class FeaturePreviewRequest(BaseModel):
    feature_type: str
    config_json: dict[str, Any] | None = None
    code: str | None = None
    dog_id: int


class FeaturePreviewResponse(BaseModel):
    value: float | None = None
    error: str | None = None
