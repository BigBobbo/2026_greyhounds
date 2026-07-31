"""Daily weather per track — Irish going is weather-driven.

One row per (track_id, date). Populated historically by
scripts/backfill_weather.py (Open-Meteo archive) and for upcoming race
days by ml.weather.ensure_weather_for_date (Open-Meteo forecast), so the
features are available BEFORE the race: same-day weather is forecastable,
which keeps them leakage-safe at serve time.
"""

from sqlalchemy import Column, Date, Float, Integer, ForeignKey, UniqueConstraint

from app.database import Base


class TrackWeather(Base):
    __tablename__ = "track_weather"
    __table_args__ = (
        UniqueConstraint("track_id", "date", name="uq_track_weather_day"),
    )

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    precip_mm = Column(Float)          # race-day precipitation total
    temp_mean_c = Column(Float)        # race-day mean temperature
    wind_max_kmh = Column(Float)       # race-day max 10m wind speed
    precip_prev48h_mm = Column(Float)  # trailing 2-day rain (track drainage)
