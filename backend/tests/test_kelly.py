"""Kelly staking tests (audit tasks D1/J7) — first coverage of the money math."""

import pytest

from app.database import SessionLocal
from app.models.bankroll import BankrollConfig
from app.services.prediction_service import (
    DEFAULT_STAKING,
    _compute_kelly_stake,
    get_staking_params,
)


def test_kelly_math_known_case():
    # p=0.4 at odds 3.5: b=2.5, f* = (2.5*0.4 - 0.6)/2.5 = 0.16
    # quarter Kelly = 0.04, below the 0.05 cap; bankroll 200 -> stake 8.00
    out = _compute_kelly_stake(0.4, 3.5, bankroll=200.0)
    assert out["bet"] is True
    assert out["full_kelly_pct"] == pytest.approx(16.0)
    assert out["stake_pct"] == pytest.approx(4.0)
    assert out["stake"] == pytest.approx(8.0)
    assert out["edge"] == pytest.approx(0.4 - 1 / 3.5, abs=1e-4)


def test_min_edge_parameter_changes_verdict():
    # p=0.3 at odds 3.5: edge ~= 0.0143 — below default 5%, above 1%
    assert _compute_kelly_stake(0.3, 3.5)["bet"] is False
    assert _compute_kelly_stake(0.3, 3.5)["reason"] == "insufficient_edge"
    assert _compute_kelly_stake(0.3, 3.5, min_edge=0.01)["bet"] is True


def test_max_stake_pct_caps_stake():
    out = _compute_kelly_stake(0.9, 3.5, bankroll=100.0, kelly_fraction=1.0)
    assert out["bet"] is True
    assert out["stake_pct"] == pytest.approx(5.0)  # capped at default 5%
    looser = _compute_kelly_stake(
        0.9, 3.5, bankroll=100.0, kelly_fraction=1.0, max_stake_pct=0.10
    )
    assert looser["stake_pct"] == pytest.approx(10.0)


def test_no_odds_means_no_bet():
    assert _compute_kelly_stake(0.5, None)["reason"] == "no_odds"
    assert _compute_kelly_stake(0.5, 1.0)["reason"] == "no_odds"


def test_params_echoed_in_result():
    out = _compute_kelly_stake(0.4, 3.5, kelly_fraction=0.5, min_edge=0.02, max_stake_pct=0.08)
    assert out["params_used"] == {
        "kelly_fraction": 0.5,
        "min_edge": 0.02,
        "max_stake_pct": 0.08,
    }


def test_get_staking_params_reads_bankroll_config():
    db = SessionLocal()
    try:
        # No config row -> defaults
        db.query(BankrollConfig).delete()
        db.commit()
        assert get_staking_params(db) == DEFAULT_STAKING

        db.add(
            BankrollConfig(
                initial_bankroll=500.0,
                current_bankroll=432.5,
                kelly_fraction=0.5,
                min_edge=0.02,
                max_stake_pct=0.10,
            )
        )
        db.commit()
        params = get_staking_params(db)
        assert params == {
            "kelly_fraction": 0.5,
            "min_edge": 0.02,
            "max_stake_pct": 0.10,
            "current_bankroll": 432.5,
        }
    finally:
        db.query(BankrollConfig).delete()
        db.commit()
        db.close()


def test_split_val_for_calibration_is_race_aligned_and_disjoint():
    """Audit C6: Optuna's objective half and the calibration half must be
    chronologically ordered, race-aligned, and disjoint."""
    import numpy as np
    import pandas as pd

    from app.services.training_service import _split_val_for_calibration

    # 6 races, 3 entries each, ascending dates
    race_ids = np.repeat([101, 102, 103, 104, 105, 106], 3)
    meta = pd.DataFrame({"race_id": race_ids})
    X = pd.DataFrame({"f": np.arange(len(race_ids))})

    halves = _split_val_for_calibration(X, meta)
    assert halves is not None
    obj_mask, cal_mask = halves
    assert not (obj_mask & cal_mask).any()
    assert (obj_mask | cal_mask).all()
    obj_races = set(race_ids[obj_mask])
    cal_races = set(race_ids[cal_mask])
    assert obj_races == {101, 102, 103}
    assert cal_races == {104, 105, 106}
    # race-aligned: no race appears in both
    assert not obj_races & cal_races


def test_split_val_returns_none_when_too_small():
    import numpy as np
    import pandas as pd

    from app.services.training_service import _split_val_for_calibration

    meta = pd.DataFrame({"race_id": np.repeat([1, 2, 3], 3)})
    X = pd.DataFrame({"f": np.arange(9)})
    assert _split_val_for_calibration(X, meta) is None
