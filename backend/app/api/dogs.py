from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.dog import Dog
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


@router.post("/", response_model=DogResponse, status_code=201)
def create_dog(dog: DogCreate, db: Session = Depends(get_db)):
    db_dog = Dog(**dog.model_dump())
    db.add(db_dog)
    db.commit()
    db.refresh(db_dog)
    return db_dog
