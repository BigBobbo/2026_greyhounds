from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Integer, String
from app.database import Base


class Dog(Base):
    __tablename__ = "dogs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    sire = Column(String)
    dam = Column(String)
    birth_date = Column(Date)
    sex = Column(String)  # 'D' (dog) / 'B' (bitch)
    colour = Column(String)
    trainer_name = Column(String, index=True)
    owner_name = Column(String)
    greyhound_data_id = Column(String, unique=True)
    gri_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
