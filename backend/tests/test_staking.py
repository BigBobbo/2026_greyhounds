"""Tests for the canonical staking module (ml/staking.py)."""

import pytest

from ml.staking import StakingConfig, allocate_daily, kelly_stake, race_kelly


CFG = StakingConfig(
    bankroll=1000.0,
    kelly_fraction=0.25,
    min_edge=0.05,
    max_stake_pct=0.05,
    commission_rate=0.0,
    min_odds=1.5,
    max_daily_exposure_pct=0.10,
)


class TestKellyStake:
    def test_refuses_missing_odds(self):
        assert kelly_stake(0.4, None, CFG) == {"bet": False, "reason": "no_odds"}

    def test_refuses_below_min_odds(self):
        out = kelly_stake(0.9, 1.3, CFG)
        assert out["bet"] is False
        assert out["reason"] == "below_min_odds"

    def test_refuses_thin_edge(self):
        # implied 1/4 = 0.25; p=0.27 -> edge 0.02 < 0.05
        out = kelly_stake(0.27, 4.0, CFG)
        assert out["bet"] is False
        assert out["reason"] == "insufficient_edge"

    def test_full_kelly_maths(self):
        # p=0.4, odds 4.0, no commission: f* = (3*0.4 - 0.6)/3 = 0.2
        out = kelly_stake(0.4, 4.0, CFG)
        assert out["bet"] is True
        assert out["full_kelly_pct"] == pytest.approx(20.0)
        # quarter Kelly 5% == cap 5% -> stake 50 on 1000
        assert out["stake"] == pytest.approx(50.0)

    def test_commission_shrinks_stake_and_can_kill_bet(self):
        no_comm = kelly_stake(0.4, 4.0, CFG)
        with_comm = kelly_stake(
            0.4, 4.0, StakingConfig(
                bankroll=1000.0, commission_rate=0.05, min_edge=0.05,
                kelly_fraction=0.25, max_stake_pct=0.05, min_odds=1.5,
            ),
        )
        assert with_comm["bet"] is True
        assert with_comm["full_kelly_pct"] < no_comm["full_kelly_pct"]

        # Marginal edge that only exists gross of commission: p=0.30 at 3.55
        # (implied 0.2817, edge 0.0183)... below min_edge anyway; use a
        # custom config with min_edge=0 to isolate the commission effect.
        cfg0 = StakingConfig(
            bankroll=1000.0, commission_rate=0.20, min_edge=0.0,
            kelly_fraction=0.25, max_stake_pct=0.05, min_odds=1.5,
        )
        # p * b_net = 0.30 * (2.55*0.8) = 0.612 < 0.70 = q -> negative Kelly
        out = kelly_stake(0.30, 3.55, cfg0)
        assert out["bet"] is False
        assert out["reason"] == "negative_expectation_after_commission"

    def test_completeness_downweights(self):
        full = kelly_stake(0.35, 4.0, CFG, completeness=1.0)
        thin = kelly_stake(0.35, 4.0, CFG, completeness=0.5)
        assert thin["bet"] and full["bet"]
        assert thin["stake"] == pytest.approx(full["stake"] * 0.5, abs=0.02)


class TestRaceKelly:
    def test_single_value_bet_matches_single_kelly(self):
        candidates = [
            {"id": 1, "win_prob": 0.40, "odds_decimal": 4.0},
            {"id": 2, "win_prob": 0.20, "odds_decimal": 4.0},  # edge -0.05: no bet
        ]
        out = race_kelly(candidates, CFG)
        solo = kelly_stake(0.40, 4.0, CFG)
        assert out[1]["stake"] == solo["stake"]
        assert out[2]["bet"] is False

    def test_two_value_dogs_solved_jointly(self):
        # Both dogs clear min_edge individually. Joint solve (Smoczynski &
        # Tomkins): order by p*o; include while er > reserve.
        #   dog1: p=0.40, o=4.0 (er 1.6); dog2: p=0.30, o=4.5 (er 1.35)
        #   include dog1: R = (1-0.4)/(1-0.25) = 0.8; er2=1.35 > 0.8 -> include
        #   R = (1-0.7)/(1-0.25-1/4.5) = 0.3/0.5278 = 0.5684
        #   f1 = 0.4 - R/4 = 0.2579; f2 = 0.3 - R/4.5 = 0.1737  (full Kelly)
        candidates = [
            {"id": 1, "win_prob": 0.40, "odds_decimal": 4.0},
            {"id": 2, "win_prob": 0.30, "odds_decimal": 4.5},
        ]
        out = race_kelly(candidates, CFG)
        assert out[1]["joint_solve"] and out[2]["joint_solve"]
        assert out[1]["full_kelly_pct"] == pytest.approx(25.79, abs=0.05)
        assert out[2]["full_kelly_pct"] == pytest.approx(17.37, abs=0.05)
        # Quarter Kelly exceeds the 5% per-bet cap for dog1 -> capped
        assert out[1]["stake"] == pytest.approx(50.0)

    def test_joint_stakes_never_exceed_solo_sum_blowup(self):
        # Sanity: sum of joint full-Kelly fractions stays below 1.
        candidates = [
            {"id": i, "win_prob": p, "odds_decimal": o}
            for i, (p, o) in enumerate([(0.35, 4.0), (0.30, 4.5), (0.20, 7.0)])
        ]
        out = race_kelly(candidates, CFG)
        total_full = sum(
            (r.get("full_kelly_pct") or 0) for r in out.values() if r.get("bet")
        )
        assert total_full < 100.0


class TestAllocateDaily:
    def test_under_cap_untouched(self):
        bets = [{"stake": 30.0}, {"stake": 40.0}]  # cap = 100
        out = allocate_daily(bets, CFG)
        assert out[0]["stake"] == 30.0
        assert "stake_scaled" not in out[0]

    def test_over_cap_scales_proportionally(self):
        bets = [{"stake": 150.0, "stake_pct": 15.0}, {"stake": 50.0, "stake_pct": 5.0}]
        out = allocate_daily(bets, CFG)  # cap = 100, total = 200 -> scale 0.5
        assert out[0]["stake"] == pytest.approx(75.0)
        assert out[1]["stake"] == pytest.approx(25.0)
        assert out[0]["stake_scaled"] is True
        assert sum(b["stake"] for b in out) == pytest.approx(100.0)
