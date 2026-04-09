import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import api from '../api/client';
import type { Experiment } from '../types/models';

export default function ExperimentDetail() {
  const { id } = useParams();
  const [exp, setExp] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = () => {
      api.get<Experiment>(`/training/experiments/${id}`).then(res => {
        setExp(res.data);
        setLoading(false);
      }).catch(() => setLoading(false));
    };
    fetch();
    const interval = setInterval(fetch, 5000);
    return () => clearInterval(interval);
  }, [id]);

  if (loading) return <p className="text-gray-500">Loading...</p>;
  if (!exp) return <p className="text-red-500">Experiment not found</p>;

  const statusColor = exp.status === 'completed' ? 'bg-green-100 text-green-700' :
    exp.status === 'running' ? 'bg-yellow-100 text-yellow-700' :
    exp.status === 'failed' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700';

  // Prepare feature importance data
  const importanceData = exp.feature_importance
    ? Object.entries(exp.feature_importance)
        .sort((a, b) => Number(b[1]) - Number(a[1]))
        .slice(0, 15)
        .map(([name, value]) => ({ name: name.length > 20 ? name.slice(0, 20) + '...' : name, value: Number(value) }))
    : [];

  // ROC data
  const rocData = exp.roc_data
    ? (exp.roc_data as any).fpr?.map((fpr: number, i: number) => ({
        fpr, tpr: (exp.roc_data as any).tpr[i],
      }))
    : [];

  // Calibration data (now nested under .calibration)
  const calRaw = (exp.calibration_data as any)?.calibration;
  const calData = calRaw?.predicted_prob
    ? calRaw.predicted_prob.map((prob: number, i: number) => ({
        predicted: prob,
        actual: calRaw.actual_freq[i],
        count: calRaw.bin_counts[i],
      }))
    : [];

  // Betting P&L data
  const bettingRaw = (exp.calibration_data as any)?.betting;
  const pnlData: { race: number; pnl: number }[] = bettingRaw?.pnl_by_race || [];

  // Confusion matrix
  const cm = exp.confusion_matrix as number[][] | null;

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <Link to="/training" className="text-gray-400 hover:text-gray-600">&larr;</Link>
        <h1 className="text-2xl font-bold">{exp.name}</h1>
        <span className={`px-2 py-0.5 rounded-full text-xs ${statusColor}`}>{exp.status}</span>
      </div>

      {exp.status === 'running' && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <p className="text-yellow-700">Training in progress... This page auto-refreshes.</p>
        </div>
      )}

      {exp.error_message && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-red-700 text-sm font-mono">{exp.error_message}</p>
        </div>
      )}

      {/* Config */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Algorithm</p>
          <p className="font-semibold">{exp.algorithm}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Target</p>
          <p className="font-semibold">{exp.target}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Features</p>
          <p className="font-semibold">{exp.feature_set?.length || 0}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Duration</p>
          <p className="font-semibold">{exp.training_duration_s ? `${exp.training_duration_s.toFixed(1)}s` : '-'}</p>
        </div>
      </div>

      {/* Metrics */}
      {exp.metrics && (
        <div className="bg-white rounded-lg shadow p-5 mb-6">
          <h2 className="font-semibold mb-3">Metrics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(exp.metrics).map(([key, val]) => (
              <div key={key} className="border rounded-md p-3">
                <p className="text-xs text-gray-500">{key}</p>
                <p className="font-mono text-lg">{typeof val === 'number' ? val.toFixed(4) : String(val)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Betting P&L */}
      {bettingRaw && (
        <div className="bg-white rounded-lg shadow p-5 mb-6">
          <h2 className="font-semibold mb-3">Betting Simulation ($1 per race)</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
            <div className="border rounded-md p-3">
              <p className="text-xs text-gray-500">Top Pick P&L</p>
              <p className={`font-mono text-xl font-bold ${bettingRaw.top_pick_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${bettingRaw.top_pick_pnl}
              </p>
              <p className="text-xs text-gray-400">{bettingRaw.top_pick_races} races, ROI {bettingRaw.top_pick_roi}%</p>
            </div>
            <div className="border rounded-md p-3">
              <p className="text-xs text-gray-500">Strike Rate</p>
              <p className="font-mono text-xl font-bold">{bettingRaw.top_pick_strike_rate}%</p>
              <p className="text-xs text-gray-400">{bettingRaw.top_pick_winners}/{bettingRaw.top_pick_races} winners</p>
            </div>
            <div className="border rounded-md p-3">
              <p className="text-xs text-gray-500">Value Bets P&L</p>
              <p className={`font-mono text-xl font-bold ${bettingRaw.value_bet_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${bettingRaw.value_bet_pnl}
              </p>
              <p className="text-xs text-gray-400">{bettingRaw.value_bet_count} bets, ROI {bettingRaw.value_bet_roi}%</p>
            </div>
            {bettingRaw.kelly_pnl !== undefined && (
              <div className="border rounded-md p-3">
                <p className="text-xs text-gray-500">Kelly Criterion P&L</p>
                <p className={`font-mono text-xl font-bold ${bettingRaw.kelly_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${bettingRaw.kelly_pnl}
                </p>
                <p className="text-xs text-gray-400">{bettingRaw.kelly_races} bets, ROI {bettingRaw.kelly_roi}%</p>
              </div>
            )}
            <div className="border rounded-md p-3">
              <p className="text-xs text-gray-500">Favourite P&L (baseline)</p>
              <p className={`font-mono text-xl font-bold ${bettingRaw.favourite_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${bettingRaw.favourite_pnl}
              </p>
              <p className="text-xs text-gray-400">ROI {bettingRaw.favourite_roi}%</p>
            </div>
          </div>
          {pnlData.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-600 mb-2">Cumulative P&L (Top Pick)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={pnlData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="race" label={{ value: 'Race #', position: 'bottom', offset: -5 }} />
                  <YAxis label={{ value: 'P&L ($)', angle: -90, position: 'left' }} />
                  <Tooltip formatter={(v: any) => [`$${v}`, 'P&L']} />
                  <Line type="monotone" dataKey="pnl" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey={() => 0} stroke="#d1d5db" strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Hyperparameters */}
      <div className="bg-white rounded-lg shadow p-5 mb-6">
        <h2 className="font-semibold mb-3">Hyperparameters</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(exp.hyperparameters).map(([key, val]) => (
            <div key={key} className="border rounded-md p-2">
              <p className="text-xs text-gray-500">{key}</p>
              <p className="font-mono text-sm">{String(val)}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Feature Importance */}
        {importanceData.length > 0 && (
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="font-semibold mb-3">Feature Importance</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={importanceData} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* ROC Curve */}
        {rocData.length > 0 && (
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="font-semibold mb-3">ROC Curve</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rocData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="fpr" label={{ value: 'FPR', position: 'bottom' }} />
                <YAxis label={{ value: 'TPR', angle: -90, position: 'left' }} />
                <Tooltip />
                <Line type="monotone" dataKey="tpr" stroke="#3b82f6" dot={false} />
                <Line type="monotone" dataKey="fpr" stroke="#d1d5db" dot={false} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Calibration */}
        {calData.length > 0 && (
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="font-semibold mb-3">Calibration Plot</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="predicted" name="Predicted" label={{ value: 'Predicted Prob', position: 'bottom' }} />
                <YAxis dataKey="actual" name="Actual" label={{ value: 'Actual Freq', angle: -90, position: 'left' }} />
                <Tooltip />
                <Scatter data={calData} fill="#3b82f6" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Confusion Matrix */}
        {cm && (
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="font-semibold mb-3">Confusion Matrix</h2>
            <div className="flex justify-center">
              <table className="border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 text-xs text-gray-400"></th>
                    <th className="p-2 text-xs text-gray-500">Pred 0</th>
                    <th className="p-2 text-xs text-gray-500">Pred 1</th>
                  </tr>
                </thead>
                <tbody>
                  {cm.map((row, i) => (
                    <tr key={i}>
                      <td className="p-2 text-xs text-gray-500 font-medium">Actual {i}</td>
                      {row.map((val, j) => (
                        <td key={j} className={`p-3 text-center font-mono border ${
                          i === j ? 'bg-blue-50 text-blue-700 font-bold' : 'bg-gray-50 text-gray-600'
                        }`}>
                          {val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
