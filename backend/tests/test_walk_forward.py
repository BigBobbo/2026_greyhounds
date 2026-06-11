"""Tests for walk-forward fold generation."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ml.dataset_builder import generate_walk_forward_fold_indices


def _make_inputs(n_races: int, dogs_per_race: int = 6, start_date=None):
    """Build race_ids and race_dates series with `dogs_per_race` entries per race."""
    start_date = start_date or date(2025, 1, 1)
    race_ids = []
    race_dates = []
    for r in range(n_races):
        for _ in range(dogs_per_race):
            race_ids.append(r + 1)
            race_dates.append(start_date + timedelta(days=r))
    race_ids_s = pd.Series(race_ids)
    race_dates_s = pd.Series(race_dates)
    return race_ids_s, race_dates_s


def test_returns_requested_number_of_folds():
    race_ids, race_dates = _make_inputs(100)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=4, embargo_days=0,
    )
    assert len(folds) == 4


def test_folds_are_chronologically_ordered():
    race_ids, race_dates = _make_inputs(80)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=3, embargo_days=0,
    )
    for train_idx, val_idx in folds:
        # Every train row's race_date must be <= every val row's race_date
        train_max = race_dates.iloc[train_idx].max()
        val_min = race_dates.iloc[val_idx].min()
        assert train_max <= val_min


def test_train_set_expands_across_folds():
    race_ids, race_dates = _make_inputs(80)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=4, embargo_days=0,
    )
    sizes = [len(tr) for tr, _ in folds]
    # Expanding window: each subsequent fold has a larger train set
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_embargo_gaps_out_val_races():
    # 100 races on consecutive days, embargo 7 days
    race_ids, race_dates = _make_inputs(100)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=3, embargo_days=7,
    )
    for train_idx, val_idx in folds:
        train_last = race_dates.iloc[train_idx].max()
        val_first = race_dates.iloc[val_idx].min()
        # val races must start at least embargo_days+1 after train end
        gap = (val_first - train_last).days
        assert gap > 7


def test_race_integrity_preserved():
    """All dogs in a race must stay in the same fold set."""
    race_ids, race_dates = _make_inputs(60, dogs_per_race=6)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=3, embargo_days=0,
    )
    for train_idx, val_idx in folds:
        train_races = set(race_ids.iloc[train_idx].unique())
        val_races = set(race_ids.iloc[val_idx].unique())
        assert train_races.isdisjoint(val_races)
        # Every row of a race should go together
        for rid in train_races:
            assert (race_ids.iloc[train_idx] == rid).sum() == 6
        for rid in val_races:
            assert (race_ids.iloc[val_idx] == rid).sum() == 6


def test_single_fold_returns_one_pair():
    race_ids, race_dates = _make_inputs(50)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=1, embargo_days=0,
    )
    assert len(folds) == 1
    tr, v = folds[0]
    assert len(tr) > 0
    assert len(v) > 0


def test_returns_empty_when_too_few_races():
    race_ids, race_dates = _make_inputs(3)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=5, embargo_days=0,
    )
    assert folds == []


def test_extreme_embargo_drops_entire_val_windows():
    # 30 races over 30 days, embargo 365 days — no fold should survive
    race_ids, race_dates = _make_inputs(30)
    folds = generate_walk_forward_fold_indices(
        race_ids, race_dates, n_folds=3, embargo_days=365,
    )
    assert folds == []


def test_descending_input_raises():
    """Audit C2: with newest-first input the embargo check silently discarded
    every fold and training fell back to a single split. The generator must
    refuse non-ascending input loudly instead."""
    import pytest

    race_ids, race_dates = _make_inputs(50)
    race_ids_desc = race_ids.iloc[::-1].reset_index(drop=True)
    race_dates_desc = race_dates.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ascending"):
        generate_walk_forward_fold_indices(
            race_ids_desc, race_dates_desc, n_folds=3, embargo_days=0,
        )


def test_group_sizes_require_contiguity():
    """Audit C2: interleaved race ids would fragment LambdaRank groups
    silently; _compute_group_sizes must fail loudly."""
    import pytest

    from ml.dataset_builder import _compute_group_sizes

    ok = pd.Series([1, 1, 1, 2, 2, 3, 3, 3])
    assert _compute_group_sizes(ok) == [3, 2, 3]

    fragmented = pd.Series([1, 1, 2, 2, 1, 3])
    with pytest.raises(ValueError, match="contiguous"):
        _compute_group_sizes(fragmented)
