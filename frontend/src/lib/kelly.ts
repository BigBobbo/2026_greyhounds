/**
 * Client-side Kelly staking — mirrors backend `_compute_kelly_stake` so live
 * edits to the market-odds input update the BET/PASS verdict instantly
 * without a server round-trip.
 *
 * The parameters are NOT constants: they come from the user's Bankroll
 * settings (`GET /bankroll/config`). Hardcoding them here previously meant a
 * user who lowered min_edge in settings still saw "PASS — need 5%".
 */

export interface KellyInfo {
  bet: boolean;
  reason?: string;
  stake?: number;
  stake_pct?: number;
  full_kelly_pct?: number;
  edge?: number;
  implied_prob?: number;
  expected_value?: number;
}

export interface StakingParams {
  kelly_fraction: number;
  min_edge: number;
  max_stake_pct: number;
}

/** Matches the backend's DEFAULT_STAKING fallback when no config row exists. */
export const DEFAULT_STAKING: StakingParams = {
  kelly_fraction: 0.25,
  min_edge: 0.05,
  max_stake_pct: 0.05,
};

export function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

export function computeKelly(
  winProb: number | null | undefined,
  oddsDecimal: number | null | undefined,
  bankroll: number,
  params: StakingParams = DEFAULT_STAKING,
): KellyInfo {
  if (winProb == null) return { bet: false, reason: 'no_probability' };
  if (oddsDecimal == null || oddsDecimal <= 1.0) {
    return { bet: false, reason: 'no_odds' };
  }
  const impliedProb = 1.0 / oddsDecimal;
  const edge = winProb - impliedProb;
  if (edge < params.min_edge) {
    return {
      bet: false,
      reason: 'insufficient_edge',
      edge: round4(edge),
      implied_prob: round4(impliedProb),
    };
  }
  const b = oddsDecimal - 1;
  const fStar = (b * winProb - (1 - winProb)) / b;
  const fractionalKelly = Math.max(0, fStar * params.kelly_fraction);
  const stakePct = Math.min(fractionalKelly, params.max_stake_pct);
  const stake = Math.round(bankroll * stakePct * 100) / 100;
  return {
    bet: true,
    stake,
    stake_pct: Math.round(stakePct * 10000) / 100,
    full_kelly_pct: Math.round(fStar * 10000) / 100,
    edge: round4(edge),
    implied_prob: round4(impliedProb),
    expected_value: round4(winProb * (oddsDecimal - 1) - (1 - winProb)),
  };
}

export function verdictReason(
  kelly: KellyInfo,
  winProb: number | null | undefined,
  odds: number | null | undefined,
  minEdge: number = DEFAULT_STAKING.min_edge,
): string {
  if (winProb == null) return 'no model probability';
  if (odds == null || odds <= 1) return 'enter market odds →';
  if (kelly.reason === 'insufficient_edge' && kelly.edge != null) {
    const edgePct = (kelly.edge * 100).toFixed(1);
    const needPct = (minEdge * 100).toFixed(1);
    return `PASS — only ${edgePct}% edge (need ${needPct}%)`;
  }
  return 'PASS';
}
