from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.feature_definition import FeatureDefinition
from app.schemas.feature import (
    FeatureDefinitionCreate,
    FeatureDefinitionResponse,
    FeatureDefinitionUpdate,
)

router = APIRouter(prefix="/features", tags=["features"])


@router.get("/", response_model=list[FeatureDefinitionResponse])
def list_features(enabled_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(FeatureDefinition)
    if enabled_only:
        query = query.filter(FeatureDefinition.enabled.is_(True))
    return query.order_by(FeatureDefinition.name).all()


@router.get("/{feature_id}", response_model=FeatureDefinitionResponse)
def get_feature(feature_id: int, db: Session = Depends(get_db)):
    feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    return feature


@router.post("/", response_model=FeatureDefinitionResponse, status_code=201)
def create_feature(feature: FeatureDefinitionCreate, db: Session = Depends(get_db)):
    db_feature = FeatureDefinition(**feature.model_dump())
    db.add(db_feature)
    db.commit()
    db.refresh(db_feature)
    return db_feature


@router.patch("/{feature_id}", response_model=FeatureDefinitionResponse)
def update_feature(feature_id: int, update: FeatureDefinitionUpdate, db: Session = Depends(get_db)):
    db_feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not db_feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    for key, value in update.model_dump(exclude_unset=True).items():
        setattr(db_feature, key, value)
    db.commit()
    db.refresh(db_feature)
    return db_feature


@router.delete("/{feature_id}", status_code=204)
def delete_feature(feature_id: int, db: Session = Depends(get_db)):
    db_feature = db.query(FeatureDefinition).filter(FeatureDefinition.id == feature_id).first()
    if not db_feature:
        raise HTTPException(status_code=404, detail="Feature not found")
    db.delete(db_feature)
    db.commit()
