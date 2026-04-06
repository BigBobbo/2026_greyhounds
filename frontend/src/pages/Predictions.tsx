import { useEffect, useState } from 'react';
import api from '../api/client';
import type { Experiment } from '../types/models';

interface RacePrediction {
  race_id: number;
  race_date: string;
  race_number: number;
  track_name: string;
  track_code?: string;
  distance_m: number;
  grade: string;
  predictions: {
    dog_name: string;
    trap: number;
    win_probability: number | null;
    predicted_position: number | null;
    predicted_time: number | null;
  }[];
}

interface ComparisonResult {
  race_date: string;
  race_number: number;
  track_name: string;
  dog_name: string;
  trap: number;
  win_probability: number | null;
  actual_position: number;
  sp_decimal: number | null;
  won: boolean;
  value: boolean | null;
}

export default function Predictions() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExp, setSelectedExp] = useState<number | null>(null);
  const [raceId, setRaceId] = useState('');
  const [predictions, setPredictions] = useState<RacePrediction | null>(null);
  const [comparisons, setComparisons] = useState<ComparisonResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<'predict' | 'compare'>('predict');

  useEffect(() => {
    api.get<Experiment[]>('/training/experiments?status=completed').then(res => {
      setExperiments(res.data);
      if (res.data.length > 0) setSelectedExp(res.data[0].id);
    });
  }, []);

  const handlePredict = async () => {
    if (!selectedExp || !raceId) return;
    setLoading(true);
    try {
      const res = await api.get(`/predictions/race/${raceId}?experiment_id=${selectedExp}`);
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
          {/* Model selector */}
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <div className="flex items-center gap-4">
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
              <div className="bg-white rounded-lg shadow p-5 mb-6">
                <h2 className="font-semibold mb-3">Predict a Race</h2>
                <div className="flex gap-3 items-end">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Race ID</label>
                    <input
                      type="number"
                      value={raceId}
                      onChange={(e) => setRaceId(e.target.value)}
                      placeholder="Enter race ID"
                      className="border rounded-md px-3 py-2 text-sm w-32"
                    />
                  </div>
                  <button
                    onClick={handlePredict}
                    disabled={loading || !raceId}
                    className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {loading ? 'Predicting...' : 'Predict'}
                  </button>
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  Find race IDs at /api/races/ — use any resulted or scheduled race
                </p>
              </div>

              {predictions && (
                <div className="bg-white rounded-lg shadow overflow-hidden">
                  <div className="px-5 pt-4 pb-2">
                    <h2 className="font-semibold">
                      {predictions.track_name} — Race {predictions.race_number}
                    </h2>
                    <p className="text-sm text-gray-500">
                      {predictions.race_date} | {predictions.distance_m}m | {predictions.grade}
                    </p>
                  </div>
                  <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                      <tr>
                        <th className="px-4 py-3">Rank</th>
                        <th className="px-4 py-3">Trap</th>
                        <th className="px-4 py-3">Dog</th>
                        <th className="px-4 py-3">Win Prob</th>
                        <th className="px-4 py-3">Pred Position</th>
                        <th className="px-4 py-3">Pred Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {predictions.predictions.map((p, i) => (
                        <tr key={i} className={i === 0 ? 'bg-green-50' : 'hover:bg-gray-50'}>
                          <td className="px-4 py-3 font-medium">{i + 1}</td>
                          <td className="px-4 py-3">{p.trap || '-'}</td>
                          <td className="px-4 py-3 font-medium">{p.dog_name}</td>
                          <td className={`px-4 py-3 font-mono ${probColor(p.win_probability)}`}>
                            {p.win_probability ? `${(p.win_probability * 100).toFixed(1)}%` : '-'}
                          </td>
                          <td className="px-4 py-3 font-mono">
                            {p.predicted_position ? p.predicted_position.toFixed(1) : '-'}
                          </td>
                          <td className="px-4 py-3 font-mono">
                            {p.predicted_time ? p.predicted_time.toFixed(2) : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
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
