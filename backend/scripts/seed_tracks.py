"""Seed the database with Irish greyhound tracks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal, engine, Base
from app.models.track import Track
import app.models  # noqa: F401

IRISH_TRACKS = [
    {
        "name": "Curraheen Park",
        "code": "CRK",
        "location": "Cork",
        "distances_m": [325, 480, 525, 550, 570, 750],
        "num_traps": 6,
    },
    {
        "name": "Drumbo Park",
        "code": "DBP",
        "location": "Belfast",
        "distances_m": [325, 480, 525, 575],
        "num_traps": 6,
    },
    {
        "name": "Dundalk Stadium",
        "code": "DLK",
        "location": "Dundalk",
        "distances_m": [325, 400, 480, 525, 550, 750],
        "num_traps": 6,
    },
    {
        "name": "Enniscorthy",
        "code": "ENN",
        "location": "Enniscorthy",
        "distances_m": [325, 480, 525, 550, 725],
        "num_traps": 6,
    },
    {
        "name": "Galway (Greenpark)",
        "code": "GAL",
        "location": "Galway",
        "distances_m": [325, 480, 525, 550, 750],
        "num_traps": 6,
    },
    {
        "name": "Kilkenny",
        "code": "KKY",
        "location": "Kilkenny",
        "distances_m": [325, 400, 480, 525, 550, 725],
        "num_traps": 6,
    },
    {
        "name": "Limerick",
        "code": "LMK",
        "location": "Limerick",
        "distances_m": [325, 400, 480, 525, 550, 750],
        "num_traps": 6,
    },
    {
        "name": "Mullingar",
        "code": "MUL",
        "location": "Mullingar",
        "distances_m": [325, 480, 525, 550, 725],
        "num_traps": 6,
    },
    {
        "name": "Newbridge",
        "code": "NBR",
        "location": "Newbridge",
        "distances_m": [325, 480, 525, 550, 725],
        "num_traps": 6,
    },
    {
        "name": "Shelbourne Park",
        "code": "SHP",
        "location": "Dublin",
        "distances_m": [325, 400, 480, 525, 550, 750],
        "num_traps": 6,
    },
    {
        "name": "Thurles",
        "code": "THL",
        "location": "Thurles",
        "distances_m": [325, 480, 525, 550, 725],
        "num_traps": 6,
    },
    {
        "name": "Tralee",
        "code": "TRA",
        "location": "Tralee",
        "distances_m": [325, 480, 525, 550, 570, 750],
        "num_traps": 6,
    },
    {
        "name": "Waterford",
        "code": "WFD",
        "location": "Waterford",
        "distances_m": [325, 460, 480, 525, 550, 680],
        "num_traps": 6,
    },
    {
        "name": "Clonmel",
        "code": "CLM",
        "location": "Clonmel",
        "distances_m": [325, 400, 480, 525, 550, 725],
        "num_traps": 6,
    },
    {
        "name": "Derry",
        "code": "DRY",
        "location": "Derry",
        "distances_m": [325, 480, 525, 550],
        "num_traps": 6,
    },
    {
        "name": "Longford",
        "code": "LFD",
        "location": "Longford",
        "distances_m": [325, 400, 480, 525],
        "num_traps": 6,
    },
    {
        "name": "Youghal",
        "code": "YGL",
        "location": "Youghal",
        "distances_m": [325, 480, 525, 550],
        "num_traps": 6,
    },
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        existing = {t.code for t in db.query(Track).all()}
        added = 0
        for track_data in IRISH_TRACKS:
            if track_data["code"] not in existing:
                db.add(Track(**track_data))
                added += 1
        db.commit()
        print(f"Seeded {added} new tracks ({len(existing)} already existed)")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
