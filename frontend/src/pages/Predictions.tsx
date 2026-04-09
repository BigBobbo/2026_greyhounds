import { useEffect, useState } from 'react';
import api from '../api/client';
import type { Experiment, Track } from '../types/models';

interface KellyInfo {
  bet: boolean;
  reason?: string;
  stake?: number;
  stake_pct?: number;
  full_kelly_pct?: number;
  edge?: number;
  implied_prob?: number;
  expected_value?: number;
}

interface PredictionEntry {
  dog_name: string;
  trap: number;
  win_probability: number | null;
  predicted_position: number | null;
  predicted_time: number | null;
  confidence: number | null;
  confidence_tier: string | null;
  margin: number | null;
  entropy: number | null;
  edge: number | null;
  is_value: boolean | null;
  kelly: KellyInfo | null;
}

interface RacePrediction {
  race_id: number;
  race_date: string;
  race_number: number;
  track_name: string;
  track_code?: string;
  distance_m: number;
  grade: string;
  predictions: PredictionEntry[];
}

interface RaceOption {
  id: number;
  race_number: number;
  distance_m: number;
  grade: string;
  status: string;
  race_date: string;
  track_name: string;
  track_code: string;
}

interface ComparisonResult {
  race_date: string;
  race_number: number;
  track_name: string;
  dog_name: string;
  trap: number;
  win_probability: number | null;
  confidence: number | null;
  actual_position: number;
  sp_decimal: number | null;
  edge: number | null;
  won: boolean;
  value: boolean | null;
}

export default function Predictions() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selectedExp, setSelectedExp] = useState<number | null>(null);
  const [predictions, setPredictions] = useState<RacePrediction | null>(null);
  const [comparisons, setComparisons] = useState<ComparisonResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<'predict' | 'compare'>('predict');

  // Race picker state
  const [raceDate, setRaceDate] = useState(new Date().toISOString().split('T')[0]);
  const [trackCode, setTrackCode] = useState('');
  const [availableRaces, setAvailableRaces] = useState<RaceOption[]>([]);
  const [selectedRaceId, setSelectedRaceId] = useState<number | null>(null);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [bankroll, setBankroll] = useState(100);

  // Manual race ID fallback
  const [manualRaceId, setManualRaceId] = useState('');

  useEffect(() => {
    Promise.all([
      api.get<Experiment[]>('/training/experiments?status=completed'),
      api.get<Track[]>('/tracks/'),
    ]).then(([expRes, trackRes]) => {
      setExperiments(expRes.data);
      setTracks(trackRes.data);
      if (expRes.data.length > 0) setSelectedExp(expRes.data[0].id);
    });
  }, []);

  // Fetch races when date/track changes
  useEffect(() => {
    if (!raceDate) return;
    setLoadingRaces(true);
    const params = new URLSearchParams({ race_date: raceDate });
    if (trackCode) params.set('track_code', trackCode);
    api.get<RaceOption[]>(`/predictions/races-for-date?${params}`)
      .then(res => {
        setAvailableRaces(res.data);
        setSelectedRaceId(null);
      })
      .catch(() => setAvailableRaces([]))
      .finally(() => setLoadingRaces(false));
  }, [raceDate, trackCode]);

  const handlePredict = async () => {
    const raceId = selectedRaceId || (manualRaceId ? parseInt(manualRaceId) : null);
    if (!selectedExp || !raceId) return;
    setLoading(true);
    try {
      const res = await api.get(
        `/predictions/race/${raceId}?experiment_id=${selectedExp}&bankroll=${bankroll}`
      );
      setPredictions(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to generate predictions');
    }
    setLoading(false);
  };

  const handleCompare = async () => {
    if (!selectedExp) return;
    setLoading(true);
    try {
      const res = await api.get(`/predictions/results-comparison?experiment_id=${selectedExp}&limit=100`);
      setComparisons(res.data.results);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to load comparisons');
    }
    setLoading(false);
  };

  const probColor = (prob: number | null) => {
    if (!prob) return '';
    if (prob > 0.3) return 'text-green-700 font-bold';
    if (prob > 0.2) return 'text-green-600';
    if (prob > 0.15) return 'text-yellow-600';
    return 'text-gray-500';
  };

  const tierBadge = (tier: string | null) => {
    if (!tier) return null;
    const styles: Record<string, string> = {
      strong: 'bg-green-100 text-green-800',
      moderate: 'bg-yellow-100 text-yellow-800',
      weak: 'bg-gray-100 text-gray-600',
      avoid: 'bg-red-100 text-red-700',
    };
    return (
      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${styles[tier] || 'bg-gray-100'}`}>
        {tier.toUpperCase()}
      </span>
    );
  };

  const edgeDisplay = (edge: number | null) => {
    if (edge === null || edge === undefined) return '-';
    const pct = (edge * 100).toFixed(1);
    if (edge > 0.05) return <span className="text-green-600 font-medium">+{pct}%</span>;
    if (edge > 0) return <span className="text-yellow-600">+{pct}%</span>;
    return <span className="text-red-500">{pct}%</span>;
  };

  // Group available races by track
  const racesByTrack: Record<string, RaceOption[]> = {};
  availableRaces.forEach(r => {
    if (!racesByTrack[r.track_name]) racesByTrack[r.track_name] = [];
    racesByTrack[r.track_name].push(r);
  });

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Predictions</h1>

      {experiments.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 text-lg">No trained models yet</p>
          <p className="text-gray-400 text-sm mt-1">Train a model in the Training Lab first</p>
        </div>
      ) : (
        <>
          {/* Model selector + bankroll */}
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <div className="flex items-center gap-4 flex-wrap">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Model</label>
                <select
                  value={selectedExp || ''}
                  onChange={(e) => setSelectedExp(parseInt(e.target.value))}
                  className="border rounded-md px-3 py-2 text-sm"
                >
                  {experiments.map(exp => (
                    <option key={exp.id} value={exp.id}>
                      {exp.name} ({exp.algorithm} / {exp.target})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Bankroll</label>
                <div className="flex items-center gap-1">
                  <span className="text-gray-400 text-sm">$</span>
                  <input
                    type="number"
                    value={bankroll}
                    onChange={(e) => setBankroll(Math.max(1, parseInt(e.target.value) || 100))}
                    className="border rounded-md px-3 py-2 text-sm w-24"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex border-b mb-6">
            <button
              onClick={() => setTab('predict')}
              className={`px-5 py-3 text-sm font-medium border-b-2 ${
                tab === 'predict' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500'
              }`}
            >
              Predict Race
            </button>
            <button
              onClick={() => { setTab('compare'); handleCompare(); }}
              className={`px-5 py-3 text-sm font-medium border-b-2 ${
                tab === 'compare' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500'
              }`}
            >
              Results Comparison
            </button>
          </div>

          {tab === 'predict' && (
            <div>
              {/* Race picker */}
              <div className="bg-white rounded-lg shadow p-5 mb-6">
                <h2 className="font-semibold mb-3">Select a Race</h2>
                <div className="flex gap-3 items-end flex-wrap mb-4">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Date</label>
                    <input
                      type="date"
                      value={raceDate}
                      onChange={(e) => setRaceDate(e.target.value)}
                      className="border rounded-md px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Track (optional)</label>
                    <select
                      value={trackCode}
                      onChange={(e) => setTrackCode(e.target.value)}
                      className="border rounded-md px-3 py-2 text-sm"
                    >
                      <option value="">All tracks</option>
                      {tracks.filter(t => t.active).map(t => (
                        <option key={t.code} value={t.code}>{t.name}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {loadingRaces ? (
                  <p className="text-gray-400 text-sm">Loading races...</p>
                ) : availableRaces.length > 0 ? (
                  <div className="space-y-3">
                    {Object.entries(racesByTrack).map(([trackName, races]) => (
                      <div key={trackName}>
                        <p className="text-xs font-medium text-gray-500 uppercase mb-1">{trackName}</p>
                        <div className="flex flex-wrap gap-2">
                          {races.map(r => (
                            <button
                              key={r.id}
                              onClick={() => { setSelectedRaceId(r.id); setManualRaceId(''); }}
                              className={`px-3 py-2 rounded-md text-sm border transition-colors ${
                                selectedRaceId === r.id
                                  ? 'bg-blue-600 text-white border-blue-600'
                                  : 'bg-white hover:bg-gray-50 border-gray-200'
                              }`}
                            >
                              R{r.race_number} - {r.distance_m}m {r.grade || ''}
                              <span className={`ml-1 text-xs ${
                                selectedRaceId === r.id ? 'text-blue-200' : 'text-gray-400'
                              }`}>
                                ({r.status})
                              </span>
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">No races found for this date</p>
                )}

                {/* Manual race ID fallback */}
                <div className="mt-4 pt-3 border-t">
                  <div className="flex gap-3 items-end">
                    <div>
                      <label className="block text-xs text-gray-500 mb-1">Or enter Race ID directly</label>
                      <input
                        type="number"
                        value={manualRaceId}
                        onChange={(e) => { setManualRaceId(e.target.value); setSelectedRaceId(null); }}
                        placeholder="Race ID"
                        className="border rounded-md px-3 py-2 text-sm w-28"
                      />
                    </div>
                    <button
                      onClick={handlePredict}
                      disabled={loading || (!selectedRaceId && !manualRaceId)}
                      className="bg-blue-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
                    >
                      {loading ? 'Predicting...' : 'Predict'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Prediction results */}
              {predictions && (
                <div className="space-y-4">
                  {/* Race header + confidence summary */}
                  <div className="bg-white rounded-lg shadow px-5 pt-4 pb-3">
                    <div className="flex items-center justify-between flex-wrap gap-3">
                      <div>
                        <h2 className="font-semibold text-lg">
                          {predictions.track_name} — Race {predictions.race_number}
                        </h2>
                        <p className="text-sm text-gray-500">
                          {predictions.race_date} | {predictions.distance_m}m | {predictions.grade}
                        </p>
                      </div>
                      {predictions.predictions[0]?.confidence_tier && (
                        <div className="flex items-center gap-3">
                          <div className="text-right">
                            <p className="text-xs text-gray-500">Model Confidence</p>
                            {tierBadge(predictions.predictions[0].confidence_tier)}
                          </div>
                          {predictions.predictions[0]?.margin != null && (
                            <div className="text-right">
                              <p className="text-xs text-gray-500">Margin</p>
                              <p className="font-mono text-sm font-medium">
                                {(predictions.predictions[0].margin * 100).toFixed(1)}pp
                              </p>
                            </div>
                          )}
                          {predictions.predictions[0]?.entropy != null && (
                            <div className="text-right">
                              <p className="text-xs text-gray-500">Entropy</p>
                              <p className="font-mono text-sm">
                                {(predictions.predictions[0].entropy * 100).toFixed(0)}%
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Predictions table */}
                  <div className="bg-white rounded-lg shadow overflow-hidden">
                    <table className="w-full text-sm text-left">
                      <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                        <tr>
                          <th className="px-4 py-3">Rank</th>
                          <th className="px-4 py-3">Trap</th>
                          <th className="px-4 py-3">Dog</th>
                          <th className="px-4 py-3">Win Prob</th>
                          <th className="px-4 py-3">Edge</th>
                          <th className="px-4 py-3">Bet?</th>
                          <th className="px-4 py-3">Stake</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y">
                        {predictions.predictions.map((p, i) => {
                          const isTopPick = i === 0;
                          const kellyBet = p.kelly?.bet;
                          return (
                            <tr
                              key={i}
                              className={
                                isTopPick && kellyBet
                                  ? 'bg-green-50'
                                  : isTopPick
                                  ? 'bg-blue-50'
                                  : kellyBet
                                  ? 'bg-green-50/50'
                                  : 'hover:bg-gray-50'
                              }
                            >
                              <td className="px-4 py-3 font-medium">{i + 1}</td>
                              <td className="px-4 py-3">{p.trap || '-'}</td>
                              <td className="px-4 py-3 font-medium">{p.dog_name}</td>
                              <td className={`px-4 py-3 font-mono ${probColor(p.win_probability)}`}>
                                {p.win_probability ? `${(p.win_probability * 100).toFixed(1)}%` : '-'}
                              </td>
                              <td className="px-4 py-3 font-mono text-sm">
                                {edgeDisplay(p.edge)}
                              </td>
                              <td className="px-4 py-3">
                                {p.is_value ? (
                                  <span className="text-green-600 font-bold text-xs">VALUE</span>
                                ) : (
                                  <span className="text-gray-300 text-xs">-</span>
                                )}
                              </td>
                              <td className="px-4 py-3 font-mono text-sm">
                                {p.kelly?.bet ? (
                                  <span className="text-green-700 font-medium">
                                    ${p.kelly.stake?.toFixed(2)}
                                    <span className="text-gray-400 text-xs ml-1">
                                      ({p.kelly.stake_pct}%)
                                    </span>
                                  </span>
                                ) : (
                                  <span className="text-gray-300">-</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Betting recommendation card */}
                  {predictions.predictions[0]?.kelly?.bet && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <h3 className="font-semibold text-green-800 mb-2">Betting Recommendation</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                        <div>
                          <p className="text-green-600 text-xs">Top Pick</p>
                          <p className="font-semibold text-green-900">
                            {predictions.predictions[0].dog_name} (Trap {predictions.predictions[0].trap})
                          </p>
                        </div>
                        <div>
                          <p className="text-green-600 text-xs">Suggested Stake</p>
                          <p className="font-mono font-semibold text-green-900">
                            ${predictions.predictions[0].kelly.stake?.toFixed(2)}
                          </p>
                          <p className="text-green-600 text-xs">
                            {predictions.predictions[0].kelly.stake_pct}% of ${bankroll} bankroll
                          </p>
                        </div>
                        <div>
                          <p className="text-green-600 text-xs">Edge Over Market</p>
                          <p className="font-mono font-semibold text-green-900">
                            {((predictions.predictions[0].edge || 0) * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-green-600 text-xs">Expected Value</p>
                          <p className="font-mono font-semibold text-green-900">
                            {predictions.predictions[0].kelly.expected_value != null
                              ? `$${(predictions.predictions[0].kelly.expected_value * (predictions.predictions[0].kelly.stake || 1)).toFixed(2)}`
                              : '-'}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {predictions.predictions[0] && !predictions.predictions[0]?.kelly?.bet && (
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                      <p className="text-gray-500 text-sm">
                        No recommended bets for this race — insufficient edge over market odds.
                        {predictions.predictions[0]?.confidence_tier === 'avoid' &&
                          ' The model has low confidence in this race outcome.'}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {tab === 'compare' && (
            <div>
              {loading ? (
                <p className="text-gray-500">Loading comparisons...</p>
              ) : comparisons.length === 0 ? (
                <div className="bg-white rounded-lg shadow p-8 text-center">
                  <p className="text-gray-500">No prediction comparisons yet</p>
                  <p className="text-gray-400 text-sm mt-1">Predict some resulted races first</p>
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                      <tr>
                        <th className="px-4 py-3">Date</th>
                        <th className="px-4 py-3">Track</th>
                        <th className="px-4 py-3">Race</th>
                        <th className="px-4 py-3">Dog</th>
                        <th className="px-4 py-3">Trap</th>
                        <th className="px-4 py-3">Win Prob</th>
                        <th className="px-4 py-3">Edge</th>
                        <th className="px-4 py-3">Actual Pos</th>
                        <th className="px-4 py-3">SP</th>
                        <th className="px-4 py-3">Value?</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {comparisons.map((c, i) => (
                        <tr key={i} className={c.won ? 'bg-green-50' : 'hover:bg-gray-50'}>
                          <td className="px-4 py-3 text-xs">{c.race_date}</td>
                          <td className="px-4 py-3">{c.track_name}</td>
                          <td className="px-4 py-3">{c.race_number}</td>
                          <td className="px-4 py-3 font-medium">{c.dog_name}</td>
                          <td className="px-4 py-3">{c.trap}</td>
                          <td className={`px-4 py-3 font-mono ${probColor(c.win_probability)}`}>
                            {c.win_probability ? `${(c.win_probability * 100).toFixed(1)}%` : '-'}
                          </td>
                          <td className="px-4 py-3 font-mono text-xs">
                            {edgeDisplay(c.edge)}
                          </td>
                          <td className="px-4 py-3">
                            <span className={c.won ? 'text-green-700 font-bold' : ''}>
                              {c.actual_position}{c.won ? ' W' : ''}
                            </span>
                          </td>
                          <td className="px-4 py-3 font-mono text-xs">
                            {c.sp_decimal ? c.sp_decimal.toFixed(2) : '-'}
                          </td>
                          <td className="px-4 py-3">
                            {c.value === true && <span className="text-green-600 font-bold">VALUE</span>}
                            {c.value === false && <span className="text-gray-400">-</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
