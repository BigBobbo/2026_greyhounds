import { describe, expect, it } from 'vitest';

import { DEFAULT_STAKING, computeKelly, verdictReason } from './kelly';

// These cases mirror backend/tests/test_kelly.py so the client-side mirror
// of _compute_kelly_stake can never drift silently.
describe('computeKelly', () => {
  it('matches the backend known case: p=0.4 @ 3.5, bankroll 200', () => {
    const out = computeKelly(0.4, 3.5, 200);
    expect(out.bet).toBe(true);
    expect(out.full_kelly_pct).toBeCloseTo(16.0, 5);
    expect(out.stake_pct).toBeCloseTo(4.0, 5);
    expect(out.stake).toBeCloseTo(8.0, 2);
    expect(out.edge).toBeCloseTo(0.4 - 1 / 3.5, 4);
  });

  it('respects the configured min_edge', () => {
    expect(computeKelly(0.3, 3.5, 100).bet).toBe(false);
    expect(computeKelly(0.3, 3.5, 100).reason).toBe('insufficient_edge');
    expect(
      computeKelly(0.3, 3.5, 100, { ...DEFAULT_STAKING, min_edge: 0.01 }).bet
    ).toBe(true);
  });

  it('caps at the configured max_stake_pct', () => {
    const capped = computeKelly(0.9, 3.5, 100, {
      kelly_fraction: 1.0,
      min_edge: 0.05,
      max_stake_pct: 0.05,
    });
    expect(capped.stake_pct).toBeCloseTo(5.0, 5);
    const looser = computeKelly(0.9, 3.5, 100, {
      kelly_fraction: 1.0,
      min_edge: 0.05,
      max_stake_pct: 0.1,
    });
    expect(looser.stake_pct).toBeCloseTo(10.0, 5);
  });

  it('returns no_odds / no_probability sentinels', () => {
    expect(computeKelly(0.5, null, 100).reason).toBe('no_odds');
    expect(computeKelly(0.5, 1.0, 100).reason).toBe('no_odds');
    expect(computeKelly(null, 3.5, 100).reason).toBe('no_probability');
  });
});

describe('verdictReason', () => {
  it('shows the configured threshold, not a hardcoded 5%', () => {
    const kelly = computeKelly(0.3, 3.5, 100, {
      ...DEFAULT_STAKING,
      min_edge: 0.02,
    });
    // p=0.3 @ 3.5 -> edge ~1.4%, below 2%
    expect(kelly.bet).toBe(false);
    expect(verdictReason(kelly, 0.3, 3.5, 0.02)).toContain('need 2.0%');
  });
});
