"""Canonical bet-sizing module — the single source of truth for staking.

Before this module, five call sites (serving, combos, backtest, and two
frontend paths) each hard-coded their own Kelly fraction / edge floor /
stake cap, none of them read the BankrollConfig the UI edits, none modelled
commission, and every bet was sized independently — including mutually
exclusive outcomes in the same race, which is mathematically invalid and
let a single day's recommendations exceed the whole bankroll.

Everything here is driven by one StakingConfig:

  * ``kelly_stake``     — one bet, fractional Kelly, commission-aware.
  * ``race_kelly``      — simultaneous Kelly across the mutually exclusive
                          outcomes of ONE race (Smoczynski & Tomkins 2010
                          closed form), which is the correct way to stake
                          when the model likes two dogs in the same race.
  * ``allocate_daily``  — portfolio cap: scales a day's recommendations
                          down so total exposure never exceeds the
                          configured share of the bankroll.

Commission model: exchange commission is charged on net winnings, so a
price of ``o`` nets ``b = (o - 1) * (1 - commission)``. All edge and Kelly
maths use the net price; the reported ``implied_prob`` stays 1/o (the
market's own scale).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

logger = None  # set lazily to avoid import cost in tight loops


@dataclass
class StakingConfig:
    bankroll: float = 100.0
    kelly_fraction: float = 0.25      # quarter Kelly
    min_edge: float = 0.05            # p - 1/odds must clear this
    max_stake_pct: float = 0.05       # per-bet cap, share of bankroll
    commission_rate: float = 0.05     # exchange commission on net winnings
    min_odds: float = 1.5             # never bet shorter than this price
    max_daily_exposure_pct: float = 0.10  # all stakes on one day, combined

    # Combos are higher variance: sizing derives from the win params.
    combo_kelly_scale: float = 0.5    # half the win Kelly fraction
    combo_min_edge_scale: float = 2.0  # double the edge floor
    combo_max_stake_scale: float = 0.4  # 40% of the win per-bet cap

    @classmethod
    def from_db(cls, db) -> "StakingConfig":
        """Load the ledger's config; missing rows/columns fall back to
        defaults so the module works on an unmigrated database too."""
        from app.models.bankroll import BankrollConfig

        row = db.query(BankrollConfig).first()
        if row is None:
            return cls()
        cfg = cls(
            bankroll=float(row.current_bankroll or 100.0),
            kelly_fraction=float(row.kelly_fraction or 0.25),
            min_edge=float(row.min_edge if row.min_edge is not None else 0.05),
            max_stake_pct=float(row.max_stake_pct or 0.05),
        )
        for attr in ("commission_rate", "min_odds", "max_daily_exposure_pct"):
            val = getattr(row, attr, None)
            if val is not None:
                cfg = replace(cfg, **{attr: float(val)})
        return cfg

    def for_combos(self) -> "StakingConfig":
        # Forecast/trio bets settle at tote dividends — the operator margin
        # is baked into the dividend, so no separate commission applies.
        return replace(
            self,
            kelly_fraction=self.kelly_fraction * self.combo_kelly_scale,
            min_edge=self.min_edge * self.combo_min_edge_scale,
            max_stake_pct=self.max_stake_pct * self.combo_max_stake_scale,
            commission_rate=0.0,
        )


def _net_b(odds_decimal: float, commission_rate: float) -> float:
    """Net winnings per unit stake at decimal odds after commission."""
    return (odds_decimal - 1.0) * (1.0 - commission_rate)


def kelly_stake(
    win_prob: float,
    odds_decimal: float | None,
    cfg: StakingConfig,
    *,
    completeness: float = 1.0,
    combo: bool = False,
) -> dict[str, Any]:
    """Fractional-Kelly stake for a single bet.

    ``completeness`` (0..1, from computed_features.data_complete) downweights
    dogs with thin scraped history — a documented intent of the schema that
    was never wired into staking.
    """
    if combo:
        cfg = cfg.for_combos()

    if odds_decimal is None or odds_decimal <= 1.0:
        return {"bet": False, "reason": "no_odds"}
    if win_prob is None or not (0.0 < win_prob < 1.0):
        return {"bet": False, "reason": "no_probability"}
    if odds_decimal < cfg.min_odds:
        return {
            "bet": False, "reason": "below_min_odds",
            "min_odds": cfg.min_odds, "odds": odds_decimal,
        }

    implied_prob = 1.0 / odds_decimal
    edge = win_prob - implied_prob
    if edge < cfg.min_edge:
        return {
            "bet": False,
            "reason": "insufficient_edge",
            "edge": round(edge, 4),
            "implied_prob": round(implied_prob, 4),
        }

    b = _net_b(odds_decimal, cfg.commission_rate)
    if b <= 0:
        return {"bet": False, "reason": "no_odds"}
    f_star = (b * win_prob - (1.0 - win_prob)) / b
    if f_star <= 0:
        return {
            "bet": False,
            "reason": "negative_expectation_after_commission",
            "edge": round(edge, 4),
            "implied_prob": round(implied_prob, 4),
        }

    completeness = min(max(completeness if completeness is not None else 1.0, 0.0), 1.0)
    stake_pct = min(f_star * cfg.kelly_fraction * completeness, cfg.max_stake_pct)
    stake = round(cfg.bankroll * stake_pct, 2)

    return {
        "bet": True,
        "stake": stake,
        "stake_pct": round(stake_pct * 100, 2),
        "full_kelly_pct": round(f_star * 100, 2),
        "edge": round(edge, 4),
        "implied_prob": round(implied_prob, 4),
        "expected_value": round(win_prob * b - (1.0 - win_prob), 4),
        "commission_rate": cfg.commission_rate,
    }


def race_kelly(
    candidates: list[dict[str, Any]],
    cfg: StakingConfig,
) -> dict[Any, dict[str, Any]]:
    """Simultaneous Kelly across one race's mutually exclusive outcomes.

    ``candidates``: [{"id": entry_id, "win_prob": p, "odds_decimal": o,
                      "completeness": optional}, ...] for ONE race.

    Per-outcome Kelly on two dogs in the same race over-stakes both — the
    events are mutually exclusive, so the optimal allocation must be solved
    jointly. This uses the Smoczynski & Tomkins closed form: order
    candidates by expected revenue p*o (net of commission), greedily grow
    the bet set while the next candidate's expected revenue exceeds the
    reserve rate R = (1 - sum p) / (1 - sum 1/o) over the current set, then
    stake f_i = p_i - R / o_i of bankroll (full Kelly), scaled by the
    configured Kelly fraction and per-bet cap.

    Only candidates that individually clear min_edge / min_odds enter the
    solve — this is a value-betting overlay, not a book-balancing exercise.

    Returns {id: kelly-dict} for every candidate (non-qualifying ones get
    their no-bet reason).
    """
    results: dict[Any, dict[str, Any]] = {}
    pool = []
    for c in candidates:
        single = kelly_stake(
            c.get("win_prob"), c.get("odds_decimal"), cfg,
            completeness=c.get("completeness", 1.0),
        )
        results[c["id"]] = single
        if single.get("bet"):
            pool.append(c)

    if len(pool) <= 1:
        return results  # zero or one value bet: single-bet Kelly is exact

    # Net-of-commission odds for the joint solve
    o_net = {c["id"]: 1.0 + _net_b(c["odds_decimal"], cfg.commission_rate) for c in pool}
    pool.sort(key=lambda c: c["win_prob"] * o_net[c["id"]], reverse=True)

    included: list[dict[str, Any]] = []
    sum_p = 0.0
    sum_inv = 0.0
    for c in pool:
        er = c["win_prob"] * o_net[c["id"]]
        denom = 1.0 - sum_inv
        if denom <= 1e-9:
            break
        reserve = (1.0 - sum_p) / denom
        if er <= reserve:
            break
        included.append(c)
        sum_p += c["win_prob"]
        sum_inv += 1.0 / o_net[c["id"]]

    denom = 1.0 - sum_inv
    if not included or denom <= 1e-9:
        return results

    reserve = (1.0 - sum_p) / denom
    for c in included:
        f_full = c["win_prob"] - reserve / o_net[c["id"]]
        if f_full <= 0:
            continue
        completeness = min(max(c.get("completeness", 1.0) or 1.0, 0.0), 1.0)
        stake_pct = min(f_full * cfg.kelly_fraction * completeness, cfg.max_stake_pct)
        r = results[c["id"]]
        r["stake_pct"] = round(stake_pct * 100, 2)
        r["stake"] = round(cfg.bankroll * stake_pct, 2)
        r["full_kelly_pct"] = round(f_full * 100, 2)
        r["joint_solve"] = True

    # Anything that made the pool but fell out of the joint frontier keeps
    # bet=True with its single-bet numbers scaled to zero — mark it no-bet.
    frontier_ids = {c["id"] for c in included}
    for c in pool:
        if c["id"] not in frontier_ids:
            results[c["id"]] = {
                "bet": False,
                "reason": "dominated_in_joint_solve",
                "edge": results[c["id"]].get("edge"),
            }

    return results


def allocate_daily(
    bets: list[dict[str, Any]],
    cfg: StakingConfig,
) -> list[dict[str, Any]]:
    """Portfolio cap over a whole day's recommendations.

    Each bet dict must carry ``stake`` (absolute). If the day's total
    exceeds ``max_daily_exposure_pct`` of bankroll, every stake is scaled
    down proportionally — fractional Kelly is scale-invariant in ordering,
    so proportional shrink preserves relative sizing while capping ruin
    risk. Adds ``stake_scaled`` flag when shrunk.
    """
    total = sum(b.get("stake") or 0.0 for b in bets)
    cap = cfg.bankroll * cfg.max_daily_exposure_pct
    if total <= cap or total <= 0:
        return bets
    scale = cap / total
    for b in bets:
        if b.get("stake"):
            b["stake"] = round(b["stake"] * scale, 2)
            if "stake_pct" in b and b["stake_pct"] is not None:
                b["stake_pct"] = round(b["stake_pct"] * scale, 2)
            b["stake_scaled"] = True
            b["daily_exposure_scale"] = round(scale, 3)
    return bets
