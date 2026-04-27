from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from app.schemas.race import RaceCreate, RaceDetailResponse, RaceEntryResponse, RaceResponse

router = APIRouter(prefix="/races", tags=["races"])


@router.get("/", response_model=list[RaceResponse])
def list_races(
    track_id: int | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    grade: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Race, Track.name.label("track_name")).join(Track)
    if track_id:
        query = query.filter(Race.track_id == track_id)
    if date_from:
        query = query.filter(Race.race_date >= date_from)
    if date_to:
        query = query.filter(Race.race_date <= date_to)
    if grade:
        query = query.filter(Race.grade == grade)
    if status:
        query = query.filter(Race.status == status)

    rows = query.order_by(Race.race_date.desc(), Race.race_number).offset(offset).limit(limit).all()

    results = []
    for race, track_name in rows:
        resp = RaceResponse.model_validate(race)
        resp.track_name = track_name
        results.append(resp)
    return results


@router.get("/{race_id}", response_model=RaceDetailResponse)
def get_race(race_id: int, db: Session = Depends(get_db)):
    row = db.query(Race, Track.name.label("track_name")).join(Track).filter(Race.id == race_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Race not found")
    race, track_name = row

    entries_rows = (
        db.query(RaceEntry, Dog.name.label("dog_name"))
        .join(Dog)
        .filter(RaceEntry.race_id == race_id)
        .order_by(RaceEntry.trap)
        .all()
    )
    entries = []
    for entry, dog_name in entries_rows:
        entry_resp = RaceEntryResponse.model_validate(entry)
        entry_resp.dog_name = dog_name
        entries.append(entry_resp)

    resp = RaceDetailResponse.model_validate(race)
    resp.track_name = track_name
    resp.entries = entries
    return resp


@router.post("/", response_model=RaceResponse, status_code=201)
def create_race(race: RaceCreate, db: Session = Depends(get_db)):
    db_race = Race(**race.model_dump())
    db.add(db_race)
    db.commit()
    db.refresh(db_race)
    return RaceResponse.model_validate(db_race)


# ----- Manual race-card entry --------------------------------------------
# Lets users hand-enter an upcoming race when scraping isn't viable
# (e.g. card not yet published, GRI URL changed).

class ManualRaceEntryIn(BaseModel):
    trap: int = Field(ge=1, le=8)
    dog_name: str = Field(min_length=1)
    trainer_name: str | None = None
    weight_kg: float | None = None
    sp_decimal: float | None = None
    starting_price: str | None = None


class ManualRaceIn(BaseModel):
    track_code: str = Field(min_length=2)
    race_date: date
    race_number: int = Field(ge=1, le=20)
    race_time: str | None = None  # "HH:MM"
    distance_m: int = Field(ge=200, le=1000)
    grade: str | None = None
    race_type: str = "flat"
    going: str | None = None
    entries: list[ManualRaceEntryIn]


class ManualRaceOut(BaseModel):
    race_id: int
    track_id: int
    track_name: str
    race_number: int
    race_date: str
    entries_created: int
    dogs_created: int
    message: str


@router.post("/manual", response_model=ManualRaceOut, status_code=201)
def create_manual_race(payload: ManualRaceIn, db: Session = Depends(get_db)):
    """
    Create an upcoming race + entries from a hand-typed card.

    Idempotent on (track, date, race_number): if the race already exists
    we update its entries (replacing any existing ones to keep the trap
    set consistent).
    """
    from scraping.db_pipeline import _parse_race_time, find_or_create_dog

    track = db.query(Track).filter(Track.code == payload.track_code).first()
    if not track:
        raise HTTPException(
            status_code=404, detail=f"Unknown track code: {payload.track_code}"
        )

    if not payload.entries:
        raise HTTPException(status_code=400, detail="At least one entry is required")

    traps = [e.trap for e in payload.entries]
    if len(set(traps)) != len(traps):
        raise HTTPException(status_code=400, detail="Duplicate trap numbers in entries")

    parsed_time = _parse_race_time(payload.race_time)

    race = (
        db.query(Race)
        .filter(
            Race.track_id == track.id,
            Race.race_date == payload.race_date,
            Race.race_number == payload.race_number,
        )
        .first()
    )

    if race is None:
        race = Race(
            track_id=track.id,
            race_date=payload.race_date,
            race_time=parsed_time,
            race_number=payload.race_number,
            distance_m=payload.distance_m,
            grade=payload.grade,
            race_type=payload.race_type,
            going=payload.going,
            num_runners=len(payload.entries),
            source="manual",
            status="scheduled",
        )
        db.add(race)
        db.flush()
    else:
        # Update non-result fields if missing or changed by user
        race.distance_m = payload.distance_m
        if payload.grade:
            race.grade = payload.grade
        if payload.going:
            race.going = payload.going
        if parsed_time:
            race.race_time = parsed_time
        race.race_type = payload.race_type
        race.num_runners = len(payload.entries)
        # Wipe existing entries so re-submission stays consistent
        db.query(RaceEntry).filter(RaceEntry.race_id == race.id).delete()
        db.flush()

    dogs_before = db.query(Dog).count()
    for e in payload.entries:
        dog = find_or_create_dog(
            db,
            e.dog_name,
            trainer_name=e.trainer_name,
        )
        entry = RaceEntry(
            race_id=race.id,
            dog_id=dog.id,
            trap=e.trap,
            weight_kg=e.weight_kg,
            sp_decimal=e.sp_decimal,
            starting_price=e.starting_price,
        )
        db.add(entry)

    db.commit()
    db.refresh(race)
    dogs_after = db.query(Dog).count()

    return ManualRaceOut(
        race_id=race.id,
        track_id=track.id,
        track_name=track.name,
        race_number=race.race_number,
        race_date=str(race.race_date),
        entries_created=len(payload.entries),
        dogs_created=max(0, dogs_after - dogs_before),
        message=(
            f"Saved race {race.race_number} at {track.name} on {race.race_date} "
            f"with {len(payload.entries)} entries."
        ),
    )
