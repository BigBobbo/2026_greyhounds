"""Unit tests for the multi-entrant ELO rating system."""

from ml.elo import EloRatings


def test_initial_rating():
    elo = EloRatings(k=24.0, initial=1500.0)
    assert elo.get(1) == 1500.0
    assert elo.get(999) == 1500.0
    assert elo.count(1) == 0


def test_two_dog_race_winner_gains_loser_loses():
    elo = EloRatings(k=24.0, initial=1500.0)
    elo.update_race([(1, 1), (2, 2)])
    r1 = elo.get(1)
    r2 = elo.get(2)
    assert r1 > 1500.0
    assert r2 < 1500.0
    # Zero-sum within the race
    assert abs((r1 - 1500.0) + (r2 - 1500.0)) < 1e-9


def test_skipping_dogs_without_finish_position():
    elo = EloRatings(k=24.0, initial=1500.0)
    # One dog has no finish position — should be ignored for rating updates
    # but still counted as a run
    elo.update_race([(1, 1), (2, None), (3, 2)])
    # Dog 1 beat dog 3, dog 2 was untouched
    assert elo.get(1) > 1500.0
    assert elo.get(2) == 1500.0
    assert elo.get(3) < 1500.0


def test_multi_entrant_bounded_update():
    """Per-pair K / (n-1) should keep a single race's update bounded."""
    elo = EloRatings(k=24.0, initial=1500.0)
    # 6-runner race with clear ordering
    results = [(i, i) for i in range(1, 7)]
    elo.update_race(results)
    # Even the winner (beats 5 others) shouldn't gain more than ~K
    gain = elo.get(1) - 1500.0
    assert 0 < gain < 24.0 + 1.0
    # And the loser shouldn't drop more than ~K
    loss = 1500.0 - elo.get(6)
    assert 0 < loss < 24.0 + 1.0


def test_ties_symmetric():
    elo = EloRatings(k=24.0, initial=1500.0)
    # Dead heat — both same position
    elo.update_race([(1, 1), (2, 1)])
    # Neither rating should move (equal ratings + tied actual = no change)
    assert abs(elo.get(1) - 1500.0) < 1e-9
    assert abs(elo.get(2) - 1500.0) < 1e-9


def test_ratings_converge_toward_true_skill():
    """A stronger dog should eventually rate higher than a weaker one."""
    elo = EloRatings(k=32.0, initial=1500.0)
    # Dog 1 beats dog 2 repeatedly
    for _ in range(50):
        elo.update_race([(1, 1), (2, 2)])
    # Difference grows but is bounded by expected-score dynamics
    assert elo.get(1) - elo.get(2) > 200.0


def test_within_race_update_order_independent():
    """Updates within a race use a snapshot, so result order shouldn't matter."""
    elo_a = EloRatings(k=24.0, initial=1500.0)
    elo_b = EloRatings(k=24.0, initial=1500.0)
    # Seed with different histories to create rating differences
    for _ in range(3):
        elo_a.update_race([(1, 1), (2, 2)])
        elo_b.update_race([(1, 1), (2, 2)])
    # Now race in different list orders
    elo_a.update_race([(1, 1), (2, 2), (3, 3)])
    elo_b.update_race([(3, 3), (1, 1), (2, 2)])
    for d in (1, 2, 3):
        assert abs(elo_a.get(d) - elo_b.get(d)) < 1e-9
