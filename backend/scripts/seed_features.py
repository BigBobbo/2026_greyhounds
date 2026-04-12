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
    # === Going-adjusted time features ===
    {
        "name": "mean_adjusted_time_last5",
        "display_name": "Mean Adjusted Time (last 5)",
        "description": "Average going-adjusted finish time over last 5 races. Normalizes for track conditions.",
        "feature_type": "visual",
        "config_json": {"metric": "adjusted_time", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["adjusted_time"],
    },
    {
        "name": "mean_adjusted_time_last5_same_dist",
        "display_name": "Mean Adjusted Time (last 5, same distance)",
        "description": "Average going-adjusted finish time over last 5 races at same distance",
        "feature_type": "visual",
        "config_json": {"metric": "adjusted_time", "aggregation": "mean", "window": {"type": "last_n", "n": 5}, "filters": {"same_distance": True}},
        "input_columns": ["adjusted_time"],
    },
    {
        "name": "best_adjusted_time_last10",
        "display_name": "Best Adjusted Time (last 10)",
        "description": "Fastest going-adjusted time in last 10 races — the dog's ceiling performance",
        "feature_type": "visual",
        "config_json": {"metric": "adjusted_time", "aggregation": "min", "window": {"type": "last_n", "n": 10}, "filters": {}},
        "input_columns": ["adjusted_time"],
    },
    {
        "name": "best_adjusted_time_last10_same_dist",
        "display_name": "Best Adjusted Time (last 10, same distance)",
        "description": "Fastest going-adjusted time in last 10 races at the same distance",
        "feature_type": "visual",
        "config_json": {"metric": "adjusted_time", "aggregation": "min", "window": {"type": "last_n", "n": 10}, "filters": {"same_distance": True}},
        "input_columns": ["adjusted_time"],
    },
    # === Exponentially weighted mean features ===
    {
        "name": "ewm_finish_time_last10",
        "display_name": "Exp. Weighted Mean Finish Time (last 10)",
        "description": "Exponentially weighted mean finish time — recent races weighted more heavily. Alpha=0.5.",
        "feature_type": "visual",
        "config_json": {"metric": "finish_time", "aggregation": "ewm", "window": {"type": "last_n", "n": 10}, "filters": {}},
        "input_columns": ["finish_time"],
    },
    {
        "name": "ewm_position_last10",
        "display_name": "Exp. Weighted Mean Position (last 10)",
        "description": "Exponentially weighted mean finish position — recent races weighted more heavily",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "ewm", "window": {"type": "last_n", "n": 10}, "filters": {}},
        "input_columns": ["finish_position"],
    },
    {
        "name": "ewm_adjusted_time_last10",
        "display_name": "Exp. Weighted Mean Adjusted Time (last 10)",
        "description": "Exponentially weighted mean going-adjusted time — combines recency with going normalization",
        "feature_type": "visual",
        "config_json": {"metric": "adjusted_time", "aggregation": "ewm", "window": {"type": "last_n", "n": 10}, "filters": {}},
        "input_columns": ["adjusted_time"],
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
    # === Trap bias features ===
    {
        "name": "trap_win_rate_at_track",
        "display_name": "Trap Win Rate at Track",
        "description": "Historical win rate for this trap number at this specific track",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    same_trap = dog_history[
        (dog_history['trap'] == race_context['trap']) &
        (dog_history['track_id'] == race_context['track_id'])
    ]
    positions = same_trap['finish_position'].dropna()
    if len(positions) < 2:
        return None
    return float((positions == 1).sum() / len(positions))
""",
        "input_columns": ["finish_position", "trap", "track_id"],
    },
    {
        "name": "trap_place_rate_at_track",
        "display_name": "Trap Place Rate at Track",
        "description": "Historical top-3 rate for this trap at this track",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    same_trap = dog_history[
        (dog_history['trap'] == race_context['trap']) &
        (dog_history['track_id'] == race_context['track_id'])
    ]
    positions = same_trap['finish_position'].dropna()
    if len(positions) < 2:
        return None
    return float((positions <= 3).sum() / len(positions))
""",
        "input_columns": ["finish_position", "trap", "track_id"],
    },
    # === Race frequency / workload ===
    {
        "name": "races_in_last_30_days",
        "display_name": "Races in Last 30 Days",
        "description": "Number of races in the past 30 days — workload indicator",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if len(dog_history) == 0:
        return 0.0
    from datetime import timedelta
    cutoff = race_context['race_date'] - timedelta(days=30)
    recent = dog_history[dog_history['race_date'] >= cutoff]
    return float(len(recent))
""",
        "input_columns": ["race_date"],
    },
    # === Grade movement ===
    {
        "name": "grade_change",
        "display_name": "Grade Change",
        "description": "Whether the dog is moving up or down in class. Positive = dropping (easier), negative = stepping up (harder).",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if len(dog_history) == 0:
        return None
    # Extract numeric grade level (e.g. A1=1, A2=2, ..., A10=10)
    import re
    current_grade = race_context.get('grade', '')
    last_grade = dog_history['grade'].iloc[-1] if 'grade' in dog_history.columns else ''
    def grade_to_num(g):
        if not g:
            return None
        m = re.search(r'(\\d+)', str(g))
        return int(m.group(1)) if m else None
    curr = grade_to_num(current_grade)
    prev = grade_to_num(last_grade)
    if curr is None or prev is None:
        return None
    return float(curr - prev)
""",
        "input_columns": ["grade"],
    },
    # === Going preference ===
    {
        "name": "going_win_rate",
        "display_name": "Going Win Rate",
        "description": "Win rate on the same going/ground conditions",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if len(dog_history) == 0 or 'going' not in dog_history.columns:
        return None
    current_going = race_context.get('going', None) if hasattr(race_context, 'get') else None
    if not current_going:
        return None
    same_going = dog_history[dog_history['going'] == current_going]
    positions = same_going['finish_position'].dropna()
    if len(positions) < 2:
        return None
    return float((positions == 1).sum() / len(positions))
""",
        "input_columns": ["going", "finish_position"],
    },
    # === Consistency / reliability ===
    {
        "name": "finish_position_stdev_last5",
        "display_name": "Position Consistency (last 5)",
        "description": "Standard deviation of finish positions — lower means more predictable",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "stdev", "window": {"type": "last_n", "n": 5}, "filters": {}},
        "input_columns": ["finish_position"],
    },
    # === Race count at this distance ===
    {
        "name": "runs_at_distance",
        "display_name": "Runs at Distance",
        "description": "Number of career races at this distance",
        "feature_type": "visual",
        "config_json": {"metric": "finish_position", "aggregation": "count", "window": {"type": "all"}, "filters": {"same_distance": True}},
        "input_columns": ["finish_position"],
    },
    # === Trouble-in-running features ===
    {
        "name": "trouble_rate_last10",
        "display_name": "Trouble in Running Rate (last 10)",
        "description": "Fraction of last 10 races where the dog encountered trouble (checked, bumped, crowded, fell, hampered). Higher = more unlucky.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    recent = dog_history.tail(10)
    comments = recent["comment"].dropna().str.lower()
    if comments.empty:
        return 0.0
    trouble_keywords = ["ck", "bmp", "crd", "fell", "hampered", "baulked", "stumbled", "crowded", "checked", "bumped"]
    trouble_count = comments.apply(lambda c: 1 if sum(1 for kw in trouble_keywords if kw in c) > 0 else 0).sum()
    return float(trouble_count) / float(len(recent))
""",
        "input_columns": ["comment"],
    },
    {
        "name": "first_bend_trouble_rate",
        "display_name": "First Bend Trouble Rate (last 10)",
        "description": "Fraction of last 10 races with trouble specifically at the first bend",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    recent = dog_history.tail(10)
    comments = recent["comment"].dropna().str.lower()
    if comments.empty:
        return 0.0
    first_bend_patterns = ["ck 1", "bmp 1", "crd 1", "crowded 1", "checked 1", "bumped 1"]
    trouble_count = comments.apply(lambda c: 1 if sum(1 for p in first_bend_patterns if p in c) > 0 else 0).sum()
    return float(trouble_count) / float(len(recent))
""",
        "input_columns": ["comment"],
    },
    # === Rest / fitness features ===
    {
        "name": "optimal_rest_window",
        "display_name": "Optimal Rest Window",
        "description": "Binary: 1.0 if 7-14 days since last race (optimal), 0.0 otherwise. Based on research showing greyhounds peak with 7-14 day rest intervals.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    last_race_date = dog_history["race_date"].max()
    current_date = race_context.get("race_date")
    if last_race_date is None or current_date is None:
        return None
    days_rest = (current_date - last_race_date).days
    return 1.0 if 7 <= days_rest <= 14 else 0.0
""",
        "input_columns": ["race_date"],
    },
    {
        "name": "rest_category",
        "display_name": "Rest Category",
        "description": "Categorized rest: 1=short (<5 days), 2=quick turnaround (5-6), 3=optimal (7-14), 4=freshened (15-28), 5=layoff (29+). Encoded as integer.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    last_race_date = dog_history["race_date"].max()
    current_date = race_context.get("race_date")
    if last_race_date is None or current_date is None:
        return None
    days = (current_date - last_race_date).days
    if days < 5:
        return 1.0
    elif days < 7:
        return 2.0
    elif days <= 14:
        return 3.0
    elif days <= 28:
        return 4.0
    else:
        return 5.0
""",
        "input_columns": ["race_date"],
    },
    # === Bayesian-smoothed rate features ===
    {
        "name": "bayesian_win_rate",
        "display_name": "Bayesian-Smoothed Win Rate",
        "description": "Win rate smoothed with Bayesian prior (Beta(1,5) prior ~ 17% base rate for 6-runner races). Prevents noisy estimates from small samples.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    positions = dog_history["finish_position"].dropna()
    if positions.empty:
        return None
    wins = (positions == 1).sum()
    total = len(positions)
    prior_alpha = 1.0
    prior_beta = 5.0
    smoothed = (wins + prior_alpha) / (total + prior_alpha + prior_beta)
    return float(smoothed)
""",
        "input_columns": ["finish_position"],
    },
    {
        "name": "bayesian_place_rate",
        "display_name": "Bayesian-Smoothed Place Rate",
        "description": "Place rate (top 3) smoothed with Bayesian prior (Beta(3,3) prior ~ 50% base rate)",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    if dog_history.empty:
        return None
    positions = dog_history["finish_position"].dropna()
    if positions.empty:
        return None
    places = (positions <= 3).sum()
    total = len(positions)
    prior_alpha = 3.0
    prior_beta = 3.0
    smoothed = (places + prior_alpha) / (total + prior_alpha + prior_beta)
    return float(smoothed)
""",
        "input_columns": ["finish_position"],
    },
    # === Seasonality features ===
    {
        "name": "race_month_sin",
        "display_name": "Race Month (sin)",
        "description": "Sine encoding of race month for cyclical seasonality — captures time-of-year effects (e.g. track conditions, daylight, temperature) without a hard boundary between December and January.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    import math
    race_date = race_context.get('race_date')
    if race_date is None:
        return None
    month = race_date.month
    return float(math.sin(2 * math.pi * month / 12))
""",
        "input_columns": ["race_date"],
    },
    {
        "name": "race_month_cos",
        "display_name": "Race Month (cos)",
        "description": "Cosine encoding of race month — paired with sin encoding so the model can learn any seasonal pattern without discontinuity at year boundaries.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    import math
    race_date = race_context.get('race_date')
    if race_date is None:
        return None
    month = race_date.month
    return float(math.cos(2 * math.pi * month / 12))
""",
        "input_columns": ["race_date"],
    },
    {
        "name": "day_of_week",
        "display_name": "Day of Week",
        "description": "Day of week as integer (0=Monday, 6=Sunday). Different days may have different race quality, field strength, or track conditions.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    race_date = race_context.get('race_date')
    if race_date is None:
        return None
    return float(race_date.weekday())
""",
        "input_columns": ["race_date"],
    },
    {
        "name": "day_of_week_sin",
        "display_name": "Day of Week (sin)",
        "description": "Sine encoding of day of week for cyclical representation — avoids hard boundary between Sunday and Monday.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    import math
    race_date = race_context.get('race_date')
    if race_date is None:
        return None
    dow = race_date.weekday()
    return float(math.sin(2 * math.pi * dow / 7))
""",
        "input_columns": ["race_date"],
    },
    {
        "name": "day_of_week_cos",
        "display_name": "Day of Week (cos)",
        "description": "Cosine encoding of day of week — paired with sin encoding for full cyclical representation.",
        "feature_type": "code",
        "code": """def compute(dog_history, race_context):
    import math
    race_date = race_context.get('race_date')
    if race_date is None:
        return None
    dow = race_date.weekday()
    return float(math.cos(2 * math.pi * dow / 7))
""",
        "input_columns": ["race_date"],
    },
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        existing = {f.name for f in db.query(FeatureDefinition).all()}
        added = 0
        errors = 0
        for feat_data in PRESET_FEATURES:
            if feat_data["name"] not in existing:
                try:
                    db.add(FeatureDefinition(**feat_data))
                    db.commit()
                    added += 1
                except Exception as e:
                    db.rollback()
                    errors += 1
                    print(f"  ERROR seeding '{feat_data['name']}': {e}", flush=True)
        print(f"Seeded {added} new features, {errors} errors ({len(existing)} already existed)", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    seed()
