from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.track import Track
from app.schemas.track import TrackCreate, TrackResponse

router = APIRouter(prefix="/tracks", tags=["tracks"])


@router.get("/", response_model=list[TrackResponse])
def list_tracks(active_only: bool = True, db: Session = Depends(get_db)):
    query = db.query(Track)
    if active_only:
        query = query.filter(Track.active.is_(True))
    return query.order_by(Track.name).all()


@router.get("/{track_id}", response_model=TrackResponse)
def get_track(track_id: int, db: Session = Depends(get_db)):
    track = db.query(Track).filter(Track.id == track_id).first()
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return track


@router.post("/", response_model=TrackResponse, status_code=201)
def create_track(track: TrackCreate, db: Session = Depends(get_db)):
    db_track = Track(**track.model_dump())
    db.add(db_track)
    db.commit()
    db.refresh(db_track)
    return db_track
