from pydantic import BaseModel


class TrackBase(BaseModel):
    name: str
    code: str
    location: str | None = None
    distances_m: list[int] | None = None
    surface: str = "sand"
    num_traps: int = 6
    active: bool = True


class TrackCreate(TrackBase):
    pass


class TrackResponse(TrackBase):
    id: int

    model_config = {"from_attributes": True}
