#!/usr/bin/env python3
"""Load the data_mirror JSONL dumps into a fresh app-schema SQLite database.

Creates every table via the SQLAlchemy metadata (same schema the app runs),
then bulk-inserts tracks, dogs, races and race_entries from the mirror.
Fields the mirror doesn't carry (sectional_time is all-NULL in production,
wide_runner, adjusted_time, ...) stay NULL for later enrichment.

Usage (from backend/ so app imports resolve):
    python3 ../data_mirror/load_mirror.py [output.db]

Default output: backend/data/greyhound_local.db
"""

import gzip
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(os.path.dirname(HERE), "backend")
sys.path.insert(0, BACKEND)

from sqlalchemy import create_engine  # noqa: E402

from app.database import Base  # noqa: E402
import app.models  # noqa: F401,E402  (registers all tables on Base)


def rows(name):
    with gzip.open(os.path.join(HERE, f"{name}.jsonl.gz"), "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        BACKEND, "data", "greyhound_local.db")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)

    engine = create_engine(f"sqlite:///{out}")
    Base.metadata.create_all(engine)
    engine.dispose()

    db = sqlite3.connect(out)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")

    def insert(table, cols, records):
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        db.executemany(sql, records)
        db.commit()
        n = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n}")

    insert("tracks",
           ["id", "name", "code", "location", "distances_m", "surface",
            "num_traps", "active"],
           [(t["id"], t["name"], t["code"], t.get("location"),
             json.dumps(t.get("distances_m")) if isinstance(t.get("distances_m"), (list, dict)) else t.get("distances_m"),
             t.get("surface"), t.get("num_traps"), t.get("active", True))
            for t in rows("tracks")])

    insert("dogs",
           ["id", "name", "sire", "dam", "birth_date", "sex", "colour",
            "trainer_name", "owner_name", "greyhound_data_id", "gri_id",
            "created_at", "updated_at"],
           [(d["id"], d["name"], d.get("sire"), d.get("dam"), d.get("birth_date"),
             d.get("sex"), d.get("colour"), d.get("trainer_name"), d.get("owner_name"),
             d.get("greyhound_data_id"), d.get("gri_id"),
             d.get("created_at"), d.get("updated_at"))
            for d in rows("dogs")])

    insert("races",
           ["id", "track_id", "race_date", "race_time", "race_number",
            "distance_m", "grade", "race_type", "prize_money", "going",
            "going_allowance", "num_runners", "status", "created_at",
            "last_scraped_at", "last_scrape_log_id"],
           [(r["id"], r["track_id"], r["race_date"], r.get("race_time"),
             r.get("race_number"), r.get("distance_m"), r.get("grade"),
             r.get("race_type"), r.get("prize_money"), r.get("going"),
             r.get("going_allowance"), r.get("num_runners"),
             r.get("status", "resulted"), r.get("created_at"),
             r.get("last_scraped_at"), r.get("last_scrape_log_id"))
            for r in rows("races")])

    insert("race_entries",
           ["id", "race_id", "dog_id", "trap", "finish_position",
            "finish_time", "sectional_time", "running_positions",
            "adjusted_time", "beaten_distance", "weight_kg",
            "starting_price", "sp_decimal", "comment"],
           [(e["id"], e["race_id"], e["dog_id"], e.get("trap"),
             e.get("finish_position"), e.get("finish_time"),
             e.get("sectional_time"), e.get("running_positions"),
             e.get("adjusted_time"), e.get("beaten_distance"),
             e.get("weight_kg"), e.get("starting_price"),
             e.get("sp_decimal"), e.get("comment"))
            for e in rows("race_entries")])

    # Weather (present once dump_local.py has run with the enrichment)
    weather_path = os.path.join(HERE, "weather.jsonl.gz")
    if os.path.exists(weather_path):
        insert("track_weather",
               ["id", "track_id", "date", "precip_mm", "temp_mean_c",
                "wind_max_kmh", "precip_prev48h_mm"],
               [(w["id"], w["track_id"], w["date"], w.get("precip_mm"),
                 w.get("temp_mean_c"), w.get("wind_max_kmh"),
                 w.get("precip_prev48h_mm"))
                for w in rows("weather")])

    db.execute("PRAGMA synchronous=NORMAL")
    db.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
