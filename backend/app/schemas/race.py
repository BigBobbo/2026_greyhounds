from datetime import date, datetime, time
from pydantic import BaseModel


class RaceEntryBase(BaseModel):
    dog_id: int
    trap: int
    finish_position: int | None = None
    finish_time: float | None = None
    sectional_time: float | None = None
    beaten_distance: float | None = None
    weight_kg: float | None = None
    starting_price: str | None = None
    sp_decimal: float | None = None
    comment: str | None = None


class RaceEntryCreate(RaceEntryBase):
    pass


class RaceEntryResponse(RaceEntryBase):
    id: int
    race_id: int
    dog_name: str | None = None  # joined from dog table

    model_config = {"from_attributes": True}


class RaceBase(BaseModel):
    track_id: int
    race_date: date
    race_time: time | None = None
    race_number: int | None = None
    distance_m: int
    grade: str | None = None
    race_type: str = "flat"
    prize_money: float | None = None
    going: str | None = None
    num_runners: int | None = None
    status: str = "scheduled"


class RaceCreate(RaceBase):
    source: str | None = None
    source_id: str | None = None


class RaceResponse(RaceBase):
    id: int
    track_name: str | None = None  # joined from track table
    created_at: datetime | None = None
    last_scraped_at: datetime | None = None
    last_scrape_log_id: int | None = None

    model_config = {"from_attributes": True}


class RaceDetailResponse(RaceResponse):
    entries: list[RaceEntryResponse] = []
