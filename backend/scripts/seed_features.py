"""Seed the database with preset feature definitions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal, engine, Base
from app.models.feature_definition import FeatureDefinition
import app.models  # noqa: F401

PRESET_FEATURES = [
    # === Time-based features ===
    {
        "name": "mean_finish_time_last5",
        "display_name": "Mean Finish Time (last 5)",
        "description": "Average finish time over the dog's last 5 races",
        "feature_type": "visual",
        "config_json": {"metric": "finish_time", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["finish_time"],
    },
    {
        "name": "min_finish_time_last10",
        "display_name": "Best Finish Time (last 10)",
        "description": "Fastest finish time over the dog's last 10 races",
        "feature_type": "visual",
        "config_json": {"metric": "finish_time", "aggregation": "min", "window": {"type": "last_n", "n": 10}, "filters": {}},
        "input_columns": ["finish_time"],
    },
    {
        "name": "mean_finish_time_last5_same_dist",
        "display_name": "Mean Finish Time (last 5, same distance)",
        "description": "Average finish time over last 5 races at the same distance",
        "feature_type": "visual",
        "config_json": {"metric": "finish_time", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {"same_distance": True}},
        "input_columns": ["finish_time"],
    },
    {
        "name": "stdev_finish_time_last5",
        "display_name": "Finish Time Consistency (last 5)",
        "description": "Standard deviation of finish time — lower means more consistent",
        "feature_type": "visual",
        "config_json": {"metric": "finish_time", "aggregation": "stdev", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["finish_time"],
    },
    {
        "name": "finish_time_trend_last5",
        "display_name": "Finish Time Trend (last 5)",
        "description": "Slope of finish time over last 5 races — negative means improving",
        "feature_type": "visual",
        "config_json": {"metric": "finish_time", "aggregation": "trend", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["finish_time"],
    },
    # === Position-based features ===
    {
        "name": "mean_position_last5",
        "display_name": "Mean Position (last 5)",
        "description": "Average finishing position over last 5 races",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["finish_position"],
    },
    {
        "name": "win_rate_last10",
        "display_name": "Win Rate (last 10)",
        "description": "Percentage of wins in last 10 races",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "win_rate", "window": {"type": "last_n", "n": 10}, "filters": {}},
        "input_columns": ["finish_position"],
    },
    {
        "name": "place_rate_last10",
        "display_name": "Place Rate (last 10)",
        "description": "Percentage of top-3 finishes in last 10 races",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "place_rate", "window": {"type": "last_n", "n": 10}, "filters": {}},
        "input_columns": ["finish_position"],
    },
    {
        "name": "win_rate_same_track",
        "display_name": "Win Rate (same track)",
        "description": "Win rate at this track across all races",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "win_rate", "window": {"type": "all"}, "filters": {"same_track": True}},
        "input_columns": ["finish_position"],
    },
    {
        "name": "win_rate_same_trap",
        "display_name": "Win Rate (same trap)",
        "description": "Win rate from this trap number",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "win_rate", "window": {"type": "all"}, "filters": {"same_trap": True}},
        "input_columns": ["finish_position"],
    },
    # === Sectional / early speed ===
    {
        "name": "mean_sectional_last5",
        "display_name": "Mean Sectional Time (last 5)",
        "description": "Average early pace (time to first bend)",
        "feature_type": "visual",
        "config_json": {"metric": "sectional_time", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["sectional_time"],
    },
    # === Weight ===
    {
        "name": "mean_weight_last5",
        "display_name": "Mean Weight (last 5)",
        "description": "Average racing weight over last 5 races",
        "feature_type": "visual",
        "config_json": {"metric": "weight_kg", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["weight_kg"],
    },
    # === Beaten distance ===
    {
        "name": "mean_beaten_dist_last5",
        "display_name": "Mean Beaten Distance (last 5)",
        "description": "Average lengths behind winner — lower is better",
        "feature_type": "visual",
        "config_json": {"metric": "beaten_distance", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["beaten_distance"],
    },
    # === SP / market features ===
    {
        "name": "mean_sp_last5",
        "display_name": "Mean SP (last 5)",
        "description": "Average starting price — indicates market assessment",
        "feature_type": "visual",
        "config_json": {"metric": "sp_decimal", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["sp_decimal"],
    },
    # === Experience ===
    {
        "name": "career_runs",
        "display_name": "Career Runs",
        "description": "Total number of races in career",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "count", "window": {"type": "all"}, "filters": {}},
        "input_columns": ["finish_position"],
    },
    {
        "name": "runs_at_track",
        "display_name": "Runs at Track",
        "description": "Number of races at this track",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "count", "window": {"type": "all"}, "filters": {"same_track": True}},
        "input_columns": ["finish_position"],
    },
    # === Code-based features (advanced) ===
    {
        "name": "days_since_last_race",
        "display_name": "Days Since Last Race",
        "description": "Rest days since last race — freshness indicator",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if len(dog_history) == 0:
        return None
    last_race = dog_history['race_date'].iloc[-1]
    current = race_context['race_date']
    delta = (current - last_race).days
    return float(delta)
""",
        "input_columns": ["race_date"],
    },
    {
        "name": "days_since_last_win",
        "display_name": "Days Since Last Win",
        "description": "Days since the dog last won a race",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    wins = dog_history[dog_history['finish_position'] == 1]
    if len(wins) == 0:
        return None
    last_win = wins['race_date'].iloc[-1]
    delta = (race_context['race_date'] - last_win).days
    return float(delta)
""",
        "input_columns": ["finish_position", "race_date"],
    },
    {
        "name": "improving_form",
        "display_name": "Improving Form (position slope)",
        "description": "Whether positions are improving (negative slope = getting better)",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    recent = dog_history.tail(5)
    positions = recent['finish_position'].dropna()
    if len(positions) < 3:
        return None
    x = np.arange(len(positions), dtype=float)
    y = positions.values.astype(float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)
""",
        "input_columns": ["finish_position"],
    },
    {
        "name": "track_distance_affinity",
        "display_name": "Track+Distance Affinity",
        "description": "Average position at this track and distance combo — lower is better",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    same = dog_history[
        (dog_history['track_id'] == race_context['track_id']) &
        (dog_history['distance_m'] == race_context['distance_m'])
    ]
    positions = same['finish_position'].dropna()
    if len(positions) == 0:
        return None
    return float(positions.mean())
""",
        "input_columns": ["finish_position", "track_id", "distance_m"],
    },
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        existing = {f.name for f in db.query(FeatureDefinition).all()}
        added = 0
        for feat_data in PRESET_FEATURES:
            if feat_data["name"] not in existing:
                db.add(FeatureDefinition(**feat_data))
                added += 1
        db.commit()
        print(f"Seeded {added} new features ({len(existing)} already existed)")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
