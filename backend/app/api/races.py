from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
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
