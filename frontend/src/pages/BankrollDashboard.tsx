import { useEffect, useState, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../api/client';

interface BankrollConfig {
  initial_bankroll: number;
  current_bankroll: number;
  kelly_fraction: number;
  min_edge: number;
  max_stake_pct: number;
}

interface BetRecord {
  id: number;
  race_entry_id: number;
  experiment_id: number;
  dog_name: string;
  track_name: string;
  race_date: string;
  race_number: number;
  trap: number;
  grade: string;
  win_probability: number | null;
  odds_decimal: number | null;
  edge: number | null;
  confidence_tier: string | null;
  stake: number;
  stake_method: string;
  bankroll_before: number | null;
  outcome: string;
  profit: number | null;
  bankroll_after: number | null;
  settled_at: string | null;
  created_at: string | null;
}

interface Summary {
  initial_bankroll: number;
  current_bankroll: number;
  total_pnl: number;
  total_pnl_pct: number;
  roi: number;
  total_bets: number;
  settled_bets: number;
  pending_bets: number;
  wins: number;
  losses: number;
  strike_rate: number;
  total_staked: number;
  avg_stake: number;
  streak: string;
  streak_type: string | null;
  cumulative_pnl: { bet: number; pnl: number; date: string; dog: string }[];
}

export default function BankrollDashboard() {
  const [config, setConfig] = useState<BankrollConfig | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [bets, setBets] = useState<BetRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [tab, setTab] = useState<'overview' | 'bets' | 'calculator'>('overview');

  // Settings form
  const [editBankroll, setEditBankroll] = useState(100);
  const [editKelly, setEditKelly] = useState(0.25);
  const [editMinEdge, setEditMinEdge] = useState(0.05);
  const [editMaxStake, setEditMaxStake] = useState(0.05);

  // Calculator
  const [calcProb, setCalcProb] = useState(0.25);
  const [calcOdds, setCalcOdds] = useState(4.0);

  const fetchData = useCallback(() => {
    Promise.all([
      api.get('/bankroll/config'),
      api.get('/bankroll/summary'),
      api.get('/bankroll/bets?limit=200'),
    ]).then(([configRes, summaryRes, betsRes]) => {
      setConfig(configRes.data);
      setSummary(summaryRes.data);
      setBets(betsRes.data);
      setEditBankroll(configRes.data.initial_bankroll);
      setEditKelly(configRes.data.kelly_fraction);
      setEditMinEdge(configRes.data.min_edge);
      setEditMaxStake(configRes.data.max_stake_pct);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSaveSettings = async () => {
    await api.put('/bankroll/config', {
      initial_bankroll: editBankroll,
      current_bankroll: editBankroll,
      kelly_fraction: editKelly,
      min_edge: editMinEdge,
      max_stake_pct: editMaxStake,
    });
    setShowSettings(false);
    fetchData();
  };

  const handleReset = async () => {
    if (!confirm('Reset bankroll and clear all bet history?')) return;
    await api.post('/bankroll/reset');
    fetchData();
  };

  // Settle with an explicit result string. The old version encoded "lost"
  // as actual_position=2, which the API's place-bet rule (position <= 2
  // wins) read as a WIN — clicking "Lost" on a place bet paid it out.
  const handleSettleBet = async (betId: number, result: 'won' | 'lost' | 'void') => {
    try {
      await api.post(`/bankroll/bets/${betId}/settle`, { result });
    } catch (err: any) {
      alert(err?.response?.data?.detail ?? 'Could not settle bet');
    }
    fetchData();
  };

  const handleReconcile = async () => {
    const res = await api.post('/bankroll/reconcile');
    const n = res.data?.settled_count ?? 0;
    if (n === 0) alert('No pending bets had results available yet.');
    fetchData();
  };

  const handleDeleteBet = async (betId: number) => {
    if (!confirm('Delete this bet and refund the stake?')) return;
    await api.delete(`/bankroll/bets/${betId}`);
    fetchData();
  };

  // Kelly calculator
  const kellyCalc = () => {
    if (calcOdds <= 1) return { stake: 0, full_kelly: 0, edge: 0, ev: 0 };
    const impliedProb = 1 / calcOdds;
    const edge = calcProb - impliedProb;
    const b = calcOdds - 1;
    const fStar = (b * calcProb - (1 - calcProb)) / b;
    const bankroll = config?.current_bankroll || 100;
    const fraction = config?.kelly_fraction || 0.25;
    const maxPct = config?.max_stake_pct || 0.05;
    const stakePct = Math.min(Math.max(fStar * fraction, 0), maxPct);
    const stake = bankroll * stakePct;
    const ev = calcProb * (calcOdds - 1) - (1 - calcProb);
    return {
      stake: Math.round(stake * 100) / 100,
      full_kelly: Math.round(fStar * 10000) / 100,
      edge: Math.round(edge * 10000) / 100,
      ev: Math.round(ev * 10000) / 100,
      stakePct: Math.round(stakePct * 10000) / 100,
    };
  };

  const calc = kellyCalc();

  if (loading) return <p className="text-gray-500">Loading...</p>;

  const pnlColor = (val: number) => val >= 0 ? 'text-green-600' : 'text-red-600';

  return (
    <div>
      <div className="flex items-center justify-between mb-4 sm:mb-6">
        <h1 className="text-xl sm:text-2xl font-bold">Bankroll</h1>
        <div className="flex gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="px-3 py-1.5 text-sm border rounded-md hover:bg-gray-50"
          >
            Settings
          </button>
          <button
            onClick={handleReset}
            className="px-3 py-1.5 text-sm text-red-600 border border-red-200 rounded-md hover:bg-red-50"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="bg-white rounded-lg shadow p-5 mb-6">
          <h2 className="font-semibold mb-3">Bankroll Settings</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Starting Bankroll ($)</label>
              <input type="number" value={editBankroll} onChange={e => setEditBankroll(parseFloat(e.target.value) || 100)}
                className="border rounded-md px-3 py-2 text-sm w-full" />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Kelly Fraction</label>
              <input type="number" step="0.05" min="0.05" max="1" value={editKelly}
                onChange={e => setEditKelly(parseFloat(e.target.value) || 0.25)}
                className="border rounded-md px-3 py-2 text-sm w-full" />
              <p className="text-xs text-gray-400 mt-0.5">0.25 = quarter Kelly (recommended)</p>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Min Edge Required</label>
              <input type="number" step="0.01" min="0" max="0.5" value={editMinEdge}
                onChange={e => setEditMinEdge(parseFloat(e.target.value) || 0.05)}
                className="border rounded-md px-3 py-2 text-sm w-full" />
              <p className="text-xs text-gray-400 mt-0.5">0.05 = 5% minimum edge</p>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Max Stake % of Bankroll</label>
              <input type="number" step="0.01" min="0.01" max="0.25" value={editMaxStake}
                onChange={e => setEditMaxStake(parseFloat(e.target.value) || 0.05)}
                className="border rounded-md px-3 py-2 text-sm w-full" />
              <p className="text-xs text-gray-400 mt-0.5">0.05 = max 5% per bet</p>
            </div>
          </div>
          <button onClick={handleSaveSettings}
            className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700">
            Save Settings
          </button>
        </div>
      )}

      {/* Summary cards */}
      {summary && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 sm:gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-xs text-gray-500">Bankroll</p>
            <p className="text-2xl font-bold font-mono">${summary.current_bankroll.toFixed(2)}</p>
            <p className={`text-xs font-medium ${pnlColor(summary.total_pnl)}`}>
              {summary.total_pnl >= 0 ? '+' : ''}{summary.total_pnl_pct}% from start
            </p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-xs text-gray-500">Total P&L</p>
            <p className={`text-2xl font-bold font-mono ${pnlColor(summary.total_pnl)}`}>
              {summary.total_pnl >= 0 ? '+' : ''}${summary.total_pnl.toFixed(2)}
            </p>
            <p className="text-xs text-gray-400">ROI: {summary.roi}%</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-xs text-gray-500">Strike Rate</p>
            <p className="text-2xl font-bold font-mono">{summary.strike_rate}%</p>
            <p className="text-xs text-gray-400">{summary.wins}W / {summary.losses}L</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-xs text-gray-500">Bets</p>
            <p className="text-2xl font-bold font-mono">{summary.total_bets}</p>
            <p className="text-xs text-gray-400">{summary.pending_bets} pending</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-xs text-gray-500">Streak</p>
            <p className={`text-2xl font-bold font-mono ${summary.streak_type === 'won' ? 'text-green-600' : summary.streak_type === 'lost' ? 'text-red-600' : ''}`}>
              {summary.streak}
            </p>
            <p className="text-xs text-gray-400">Avg stake: ${summary.avg_stake}</p>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex border-b mb-6">
        {(['overview', 'bets', 'calculator'] as const).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-5 py-3 text-sm font-medium border-b-2 capitalize ${
              tab === t ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500'
            }`}>{t}</button>
        ))}
      </div>

      {/* Overview tab */}
      {tab === 'overview' && summary && (
        <div>
          {summary.cumulative_pnl.length > 0 ? (
            <div className="bg-white rounded-lg shadow p-5">
              <h2 className="font-semibold mb-3">Cumulative P&L</h2>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={summary.cumulative_pnl}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bet" label={{ value: 'Bet #', position: 'bottom', offset: -5 }} />
                  <YAxis label={{ value: 'P&L ($)', angle: -90, position: 'left' }} />
                  <Tooltip
                    formatter={(v: any) => [`$${v}`, 'P&L']}
                    labelFormatter={(label: any) => {
                      const item = summary.cumulative_pnl[label - 1];
                      return item ? `${item.dog} (${item.date})` : `Bet #${label}`;
                    }}
                  />
                  <Line type="monotone" dataKey="pnl" stroke="#3b82f6" strokeWidth={2} dot={summary.cumulative_pnl.length < 50} />
                  <Line type="monotone" dataKey={() => 0} stroke="#d1d5db" strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow p-8 text-center">
              <p className="text-gray-500 text-lg">No settled bets yet</p>
              <p className="text-gray-400 text-sm mt-1">Go to Predictions, predict a race, and place a bet to get started</p>
            </div>
          )}
        </div>
      )}

      {/* Bets tab */}
      {tab === 'bets' && (
        <div>
          <div className="flex justify-end mb-2">
            <button
              onClick={handleReconcile}
              className="text-sm px-3 py-1.5 rounded bg-blue-600 text-white hover:bg-blue-700"
              title="Auto-settle pending bets from scraped race results"
            >
              Sync results
            </button>
          </div>
          {bets.length === 0 ? (
            <div className="bg-white rounded-lg shadow p-8 text-center">
              <p className="text-gray-500">No bets recorded yet</p>
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
              <table className="w-full text-sm text-left min-w-[720px]">
                <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                  <tr>
                    <th className="px-3 sm:px-4 py-3">Date</th>
                    <th className="px-3 sm:px-4 py-3">Race</th>
                    <th className="px-3 sm:px-4 py-3">Dog</th>
                    <th className="px-3 sm:px-4 py-3">Odds</th>
                    <th className="px-3 sm:px-4 py-3">Edge</th>
                    <th className="px-3 sm:px-4 py-3">Stake</th>
                    <th className="px-3 sm:px-4 py-3">Result</th>
                    <th className="px-3 sm:px-4 py-3">P&L</th>
                    <th className="px-3 sm:px-4 py-3">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {bets.map(b => (
                    <tr key={b.id} className={
                      b.outcome === 'won' ? 'bg-green-50' :
                      b.outcome === 'lost' ? 'bg-red-50/30' :
                      'hover:bg-gray-50'
                    }>
                      <td className="px-4 py-3 text-xs">{b.race_date}</td>
                      <td className="px-4 py-3 text-xs">
                        {b.track_name} R{b.race_number}
                        <span className="text-gray-400 ml-1">{b.grade}</span>
                      </td>
                      <td className="px-4 py-3 font-medium">
                        {b.dog_name}
                        <span className="text-gray-400 text-xs ml-1">T{b.trap}</span>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">{b.odds_decimal?.toFixed(2) || '-'}</td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {b.edge != null ? (
                          <span className={b.edge > 0 ? 'text-green-600' : 'text-red-500'}>
                            {(b.edge * 100).toFixed(1)}%
                          </span>
                        ) : '-'}
                      </td>
                      <td className="px-4 py-3 font-mono">${b.stake.toFixed(2)}</td>
                      <td className="px-4 py-3">
                        {b.outcome === 'pending' ? (
                          <span className="px-2 py-0.5 rounded-full text-xs bg-yellow-100 text-yellow-700">Pending</span>
                        ) : b.outcome === 'won' ? (
                          <span className="px-2 py-0.5 rounded-full text-xs bg-green-100 text-green-700">Won</span>
                        ) : (
                          <span className="px-2 py-0.5 rounded-full text-xs bg-red-100 text-red-700">Lost</span>
                        )}
                      </td>
                      <td className={`px-4 py-3 font-mono font-medium ${
                        b.profit != null ? pnlColor(b.profit) : ''
                      }`}>
                        {b.profit != null ? `${b.profit >= 0 ? '+' : ''}$${b.profit.toFixed(2)}` : '-'}
                      </td>
                      <td className="px-4 py-3">
                        {b.outcome === 'pending' && (
                          <div className="flex gap-1">
                            <button onClick={() => handleSettleBet(b.id, 'won')}
                              className="text-xs text-green-600 hover:underline">Won</button>
                            <span className="text-gray-300">|</span>
                            <button onClick={() => handleSettleBet(b.id, 'lost')}
                              className="text-xs text-red-500 hover:underline">Lost</button>
                            <span className="text-gray-300">|</span>
                            <button onClick={() => handleSettleBet(b.id, 'void')}
                              className="text-xs text-gray-500 hover:underline">Void</button>
                            <span className="text-gray-300">|</span>
                            <button onClick={() => handleDeleteBet(b.id)}
                              className="text-xs text-gray-400 hover:underline">Del</button>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Calculator tab */}
      {tab === 'calculator' && (
        <div className="max-w-lg">
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="font-semibold mb-4">Kelly Criterion Calculator</h2>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Your Win Probability</label>
                <input type="number" step="0.01" min="0.01" max="0.99" value={calcProb}
                  onChange={e => setCalcProb(parseFloat(e.target.value) || 0.25)}
                  className="border rounded-md px-3 py-2 text-sm w-full" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Decimal Odds</label>
                <input type="number" step="0.1" min="1.01" value={calcOdds}
                  onChange={e => setCalcOdds(parseFloat(e.target.value) || 4.0)}
                  className="border rounded-md px-3 py-2 text-sm w-full" />
              </div>
            </div>

            <div className="bg-gray-50 rounded-md p-4 space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Implied Probability</span>
                <span className="font-mono text-sm">{(100 / calcOdds).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Your Edge</span>
                <span className={`font-mono text-sm font-medium ${calc.edge > 0 ? 'text-green-600' : 'text-red-500'}`}>
                  {calc.edge > 0 ? '+' : ''}{calc.edge}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Expected Value (per $1)</span>
                <span className={`font-mono text-sm ${calc.ev > 0 ? 'text-green-600' : 'text-red-500'}`}>
                  {calc.ev > 0 ? '+' : ''}{calc.ev}%
                </span>
              </div>
              <div className="border-t pt-3 flex justify-between">
                <span className="text-sm text-gray-600">Full Kelly</span>
                <span className="font-mono text-sm">{calc.full_kelly}% of bankroll</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">
                  Fractional Kelly ({((config?.kelly_fraction || 0.25) * 100).toFixed(0)}%)
                </span>
                <span className="font-mono text-sm">{calc.stakePct}% of bankroll</span>
              </div>
              <div className="border-t pt-3 flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700">Recommended Stake</span>
                <span className="font-mono text-xl font-bold text-blue-600">
                  ${calc.stake.toFixed(2)}
                </span>
              </div>
              <p className="text-xs text-gray-400">
                Based on ${config?.current_bankroll.toFixed(2)} bankroll, {((config?.kelly_fraction || 0.25) * 100).toFixed(0)}% Kelly,{' '}
                {((config?.min_edge || 0.05) * 100).toFixed(0)}% min edge, {((config?.max_stake_pct || 0.05) * 100).toFixed(0)}% max stake
              </p>
            </div>

            {calc.edge <= (config?.min_edge || 0.05) * 100 && (
              <div className="mt-3 bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-700">
                Edge ({calc.edge}%) is below the minimum threshold ({((config?.min_edge || 0.05) * 100).toFixed(0)}%). No bet recommended.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
