from sqlalchemy import Boolean, Column, Integer, String, JSON
from app.database import Base


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    code = Column(String, unique=True, nullable=False)
    location = Column(String)
    distances_m = Column(JSON)  # [480, 525, 550, 750]
    surface = Column(String, default="sand")
    num_traps = Column(Integer, default=6)
    timezone = Column(String, default="Europe/Dublin")
    active = Column(Boolean, default=True)
