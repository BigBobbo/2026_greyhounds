#!/usr/bin/env python3
"""Dump the (enriched) local app-schema DB back into data_mirror JSONL.gz.

Run after enrichment passes (dog-profile backfill, weather backfill) so the
scraped sectionals, running positions, birth dates, trainers, going
allowances and weather survive the ephemeral container:

    python3 data_mirror/dump_local.py

Overwrites dogs/races/race_entries/tracks jsonl.gz in place (superset of
the original production-API fields) and adds weather.jsonl.gz.
load_mirror.py restores everything.
"""

import gzip
import json
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(HERE), "backend", "data", "greyhound_local.db")

TABLES = {
    "tracks": ["id", "name", "code", "location", "distances_m", "surface",
               "num_traps", "active"],
    "dogs": ["id", "name", "sire", "dam", "birth_date", "sex", "colour",
             "trainer_name", "owner_name", "greyhound_data_id", "gri_id",
             "created_at", "updated_at"],
    "races": ["id", "track_id", "race_date", "race_time", "race_number",
              "distance_m", "grade", "race_type", "prize_money", "going",
              "going_allowance", "num_runners", "status", "created_at",
              "last_scraped_at", "last_scrape_log_id"],
    "race_entries": ["id", "race_id", "dog_id", "trap", "finish_position",
                     "finish_time", "sectional_time", "running_positions",
                     "adjusted_time", "beaten_distance", "weight_kg",
                     "starting_price", "sp_decimal", "comment"],
    "weather": ["id", "track_id", "date", "precip_mm", "temp_mean_c",
                "wind_max_kmh", "precip_prev48h_mm"],
}
SOURCE_TABLE = {"weather": "track_weather"}


def main() -> None:
    db = sqlite3.connect(DB)
    for name, cols in TABLES.items():
        src = SOURCE_TABLE.get(name, name)
        path = os.path.join(HERE, f"{name}.jsonl.gz")
        n = 0
        with gzip.open(path, "wt") as f:
            for row in db.execute(f"SELECT {','.join(cols)} FROM {src}"):
                f.write(json.dumps(dict(zip(cols, row))) + "\n")
                n += 1
        print(f"{name}: {n} rows -> {path}")
    db.close()


if __name__ == "__main__":
    main()
