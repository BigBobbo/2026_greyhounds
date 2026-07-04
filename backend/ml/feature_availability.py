"""
Pre-race vs post-race feature classification.

Some features the trainer learns from depend on data GRI Ireland only
publishes after a race has run (finish times, sectional times, weights,
starting prices, going). At training time these are all populated; on an
upcoming race card they are NULL. Without an explicit guard the prediction
pipeline silently imputes them with a training-set median, producing a
distribution shift between train and serve.

This module is the single source of truth for which feature names are
post-race-only. The prediction service uses it to:

  1. Refuse to serve a model on a scheduled race when the trained feature
     set contains post-race-only features.
  2. Drive the /predictions/preflight diagnostic endpoint.
"""

from __future__ import annotations


# Feature names known to require post-race data to be meaningful.
# Each entry is documented with the field on RaceEntry / Race that drives
# the value and which is unavailable until the race has been run.
POST_RACE_FEATURE_NAMES: dict[str, str] = {
    # Weight change vs recent average — needs RaceEntry.weight_kg for the
    # current entry, which is set at the on-the-night weigh-in.
    "weight_change": "RaceEntry.weight_kg (weigh-in is on the night)",
    # Current weight as a fraction of career average — same current-entry
    # weigh-in dependency as weight_change.
    "weight_pct_of_career_avg": "RaceEntry.weight_kg (weigh-in is on the night)",
    # Going-conditional trap bias — needs Race.going for the current race,
    # which GRI only reports with the results.
    "trap_bias_deviation_going": "Race.going (reported with results)",
    # Starting-price-derived features (opt-in via include_sp_features).
    # GRI only publishes SP on the results page.
    "current_sp_decimal": "RaceEntry.sp_decimal (post-race only)",
    "current_sp_implied_prob": "RaceEntry.sp_decimal (post-race only)",
    "current_sp_log_odds": "RaceEntry.sp_decimal (post-race only)",
    "current_sp_devigged_prob": "RaceEntry.sp_decimal (post-race only)",
    "sp_rank_in_field": "RaceEntry.sp_decimal (post-race only)",
    "market_overround": "RaceEntry.sp_decimal (post-race only)",
    "is_favorite": "RaceEntry.sp_decimal (post-race only)",
    "is_second_favorite": "RaceEntry.sp_decimal (post-race only)",
    "fav_gap": "RaceEntry.sp_decimal (post-race only)",
    "second_fav_gap": "RaceEntry.sp_decimal (post-race only)",
    "sp_vs_field_mean": "RaceEntry.sp_decimal (post-race only)",
    # Odds-snapshot features depend on a live pre-race odds feed populating
    # the odds_snapshots table for both training and prediction races. The
    # scraper does not currently maintain this table; if it ever starts
    # being populated for resulted-only races, the values are real at train
    # time and zero at predict time.
    "opening_to_sp_drift": "odds_snapshots table (live odds feed)",
    "odds_steam_rate": "odds_snapshots table (live odds feed)",
    "cross_book_disagreement": "odds_snapshots table (live odds feed)",
}


def post_race_features_in_use(trained_feature_names: list[str]) -> dict[str, str]:
    """Return the subset of `trained_feature_names` that are post-race-only.

    Result maps feature_name -> human-readable reason.
    """
    return {
        name: POST_RACE_FEATURE_NAMES[name]
        for name in trained_feature_names
        if name in POST_RACE_FEATURE_NAMES
    }


class PredictionDataError(ValueError):
    """Raised when prediction is attempted with missing required data.

    Distinct from a generic ValueError so API layers can map it to a 422
    response and so test code can assert on the specific failure mode.
    """
