from datetime import date, datetime
from pydantic import BaseModel


class DogBase(BaseModel):
    name: str
    sire: str | None = None
    dam: str | None = None
    birth_date: date | None = None
    sex: str | None = None
    colour: str | None = None
    trainer_name: str | None = None
    owner_name: str | None = None


class DogCreate(DogBase):
    greyhound_data_id: str | None = None
    gri_id: str | None = None


class DogResponse(DogBase):
    id: int
    greyhound_data_id: str | None = None
    gri_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}
