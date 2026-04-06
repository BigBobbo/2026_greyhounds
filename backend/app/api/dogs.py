from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.dog import Dog
from app.models.race import Race
from app.models.race_entry import RaceEntry
from app.models.track import Track
from app.schemas.dog import DogCreate, DogResponse

router = APIRouter(prefix="/dogs", tags=["dogs"])


@router.get("/", response_model=list[DogResponse])
def list_dogs(
    search: str | None = None,
    trainer: str | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Dog)
    if search:
        query = query.filter(Dog.name.ilike(f"%{search}%"))
    if trainer:
        query = query.filter(Dog.trainer_name.ilike(f"%{trainer}%"))
    return query.order_by(Dog.name).offset(offset).limit(limit).all()


@router.get("/{dog_id}", response_model=DogResponse)
def get_dog(dog_id: int, db: Session = Depends(get_db)):
    dog = db.query(Dog).filter(Dog.id == dog_id).first()
    if not dog:
        raise HTTPException(status_code=404, detail="Dog not found")
    return dog


@router.get("/{dog_id}/form")
def get_dog_form(dog_id: int, limit: int = Query(default=30, le=100), db: Session = Depends(get_db)):
    """Get a dog's race form (history of results)."""
    dog = db.query(Dog).filter(Dog.id == dog_id).first()
    if not dog:
        raise HTTPException(status_code=404, detail="Dog not found")

    rows = (
        db.query(
            RaceEntry.trap,
            RaceEntry.finish_position,
            RaceEntry.finish_time,
            RaceEntry.sectional_time,
            RaceEntry.beaten_distance,
            RaceEntry.weight_kg,
            RaceEntry.starting_price,
            RaceEntry.sp_decimal,
            RaceEntry.comment,
            RaceEntry.grade_at_entry,
            Race.id.label("race_id"),
            Race.race_date,
            Race.race_number,
            Race.distance_m,
            Race.grade,
            Race.going,
            Track.name.label("track_name"),
            Track.code.label("track_code"),
        )
        .join(Race, RaceEntry.race_id == Race.id)
        .join(Track, Race.track_id == Track.id)
        .filter(RaceEntry.dog_id == dog_id)
        .order_by(Race.race_date.desc())
        .limit(limit)
        .all()
    )

    # Compute stats
    total_runs = len(rows)
    wins = sum(1 for r in rows if r.finish_position == 1)
    places = sum(1 for r in rows if r.finish_position and r.finish_position <= 3)
    times = [r.finish_time for r in rows if r.finish_time]
    avg_time = sum(times) / len(times) if times else None
    best_time = min(times) if times else None

    form = []
    for r in rows:
        form.append({
            "race_id": r.race_id,
            "race_date": str(r.race_date),
            "race_number": r.race_number,
            "track_name": r.track_name,
            "track_code": r.track_code,
            "distance_m": r.distance_m,
            "grade": r.grade or r.grade_at_entry,
            "going": r.going,
            "trap": r.trap,
            "finish_position": r.finish_position,
            "finish_time": r.finish_time,
            "beaten_distance": r.beaten_distance,
            "weight_kg": r.weight_kg,
            "starting_price": r.starting_price,
            "sp_decimal": r.sp_decimal,
            "comment": r.comment,
        })

    return {
        "dog": DogResponse.model_validate(dog),
        "stats": {
            "total_runs": total_runs,
            "wins": wins,
            "places": places,
            "win_pct": round(wins / total_runs * 100, 1) if total_runs > 0 else 0,
            "place_pct": round(places / total_runs * 100, 1) if total_runs > 0 else 0,
            "avg_time": round(avg_time, 2) if avg_time else None,
            "best_time": round(best_time, 2) if best_time else None,
        },
        "form": form,
    }


@router.post("/", response_model=DogResponse, status_code=201)
def create_dog(dog: DogCreate, db: Session = Depends(get_db)):
    db_dog = Dog(**dog.model_dump())
    db.add(db_dog)
    db.commit()
    db.refresh(db_dog)
    return db_dog
