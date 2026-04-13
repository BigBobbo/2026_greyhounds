import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import type { Experiment, FeatureDefinition } from '../types/models';

interface FeatureVersion {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
  feature_count?: number;
}

const ALGORITHMS = [
  { value: 'lambdarank', label: 'LambdaRank (Recommended)' },
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'logistic_regression', label: 'Logistic Regression' },
];

const TARGETS = [
  { value: 'win_prob', label: 'Win Probability' },
  { value: 'finish_position', label: 'Finish Position' },
  { value: 'finish_time', label: 'Finish Time' },
];

export default function TrainingLab() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [features, setFeatures] = useState<FeatureDefinition[]>([]);
  const [versions, setVersions] = useState<FeatureVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [creating, setCreating] = useState(false);

  // Form state
  const [name, setName] = useState('');
  const [algorithm, setAlgorithm] = useState('lambdarank');
  const [target, setTarget] = useState('win_prob');
  const [selectedFeatures, setSelectedFeatures] = useState<number[]>([]);
  const [selectedVersionId, setSelectedVersionId] = useState<number | null>(null);
  const [autoTune, setAutoTune] = useState(false);
  const [optunTrials, setOptunTrials] = useState(50);

  // Date range for train/test split
  const [testAfter, setTestAfter] = useState('2026-01-01');

  const fetchData = () => {
    Promise.all([
      api.get<Experiment[]>('/training/experiments'),
      api.get<FeatureDefinition[]>('/features/?enabled_only=true'),
      api.get<FeatureVersion[]>('/features/versions'),
    ]).then(([expRes, featRes, verRes]) => {
      setExperiments(expRes.data);
      setFeatures(featRes.data);
      setVersions(verRes.data);
      if (featRes.data.length > 0 && selectedFeatures.length === 0) {
        setSelectedFeatures(featRes.data.map(f => f.id));
      }
      setLoading(false);
    }).catch(() => setLoading(false));
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleCreate = async () => {
    if (!name.trim()) return alert('Please enter a name');
    if (selectedFeatures.length === 0) return alert('Please select at least one feature');

    setCreating(true);
    try {
      const splitConfig: Record<string, any> = {
        test_after: testAfter,
        val_pct: 0.15,
      };
      if (selectedVersionId !== null) {
        splitConfig.version_id = selectedVersionId;
      }

      await api.post('/training/experiments', {
        name,
        algorithm,
        target,
        hyperparameters: {},
        feature_set: selectedFeatures,
        split_config: splitConfig,
        auto_tune: autoTune,
        optuna_trials: optunTrials,
      });
      setShowForm(false);
      setName('');
      fetchData();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to create experiment');
    }
    setCreating(false);
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this experiment?')) return;
    await api.delete(`/training/experiments/${id}`);
    fetchData();
  };

  const statusColor = (s: string) => {
    switch (s) {
      case 'completed': return 'bg-green-100 text-green-700';
      case 'running': return 'bg-yellow-100 text-yellow-700';
      case 'failed': return 'bg-red-100 text-red-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  const STAGE_LABELS: Record<string, string> = {
    starting: 'Starting',
    building_dataset: 'Building dataset',
    training_model: 'Training model',
    evaluating: 'Evaluating',
    calibrating: 'Calibrating',
    computing_shap: 'Computing SHAP',
    saving_model: 'Saving model',
    retraining_best: 'Retraining best',
  };

  const formatStage = (stage: string | null) => {
    if (!stage) return null;
    // Handle optuna trial stages like "optuna_trial_3_of_50"
    const optunaMatch = stage.match(/^optuna_trial_(\d+)_of_(\d+)$/);
    if (optunaMatch) return `Optuna trial ${optunaMatch[1]}/${optunaMatch[2]}`;
    return STAGE_LABELS[stage] || stage;
  };

  const heartbeatAgeSeconds = (heartbeat: string | null) => {
    if (!heartbeat) return null;
    return Math.floor((Date.now() - new Date(heartbeat + 'Z').getTime()) / 1000);
  };

  const formatHeartbeatAge = (seconds: number) => {
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  const STALE_THRESHOLD_S = 120;

  const primaryMetric = (exp: Experiment) => {
    if (!exp.metrics) return '-';
    const pnl = exp.metrics['betting_top_pick_pnl'];
    if (pnl !== undefined) {
      const roi = exp.metrics['betting_top_pick_roi'];
      return `P&L: $${Number(pnl).toFixed(0)} (${Number(roi).toFixed(1)}% ROI)`;
    }
    if (exp.target === 'win_prob') {
      const auc = exp.metrics['test_roc_auc'];
      return auc ? `AUC: ${Number(auc).toFixed(4)}` : '-';
    } else if (exp.target === 'finish_time') {
      const rmse = exp.metrics['test_rmse'];
      return rmse ? `RMSE: ${Number(rmse).toFixed(3)}` : '-';
    } else {
      const acc = exp.metrics['test_accuracy'];
      return acc ? `Acc: ${Number(acc).toFixed(4)}` : '-';
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4 sm:mb-6">
        <h1 className="text-xl sm:text-2xl font-bold">Training Lab</h1>
        <button
          onClick={() => setShowForm(!showForm)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700"
        >
          {showForm ? 'Cancel' : 'New Experiment'}
        </button>
      </div>

      {/* New Experiment Form */}
      {showForm && (
        <div className="bg-white rounded-lg shadow p-5 mb-6">
          <h2 className="font-semibold mb-4">Create New Experiment</h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. XGBoost Win Prob — train 2022-2025, test 2026"
                className="border rounded-md px-3 py-2 text-sm w-full"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Algorithm</label>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                className="border rounded-md px-3 py-2 text-sm w-full"
              >
                {ALGORITHMS.map(a => <option key={a.value} value={a.value}>{a.label}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Target</label>
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="border rounded-md px-3 py-2 text-sm w-full"
              >
                {TARGETS.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                Test Set Starts After
              </label>
              <input
                type="date"
                value={testAfter}
                onChange={(e) => setTestAfter(e.target.value)}
                className="border rounded-md px-3 py-2 text-sm w-full"
              />
              <p className="text-xs text-gray-400 mt-1">
                Trains on data before this date, tests on data after.
                Betting P&L is evaluated on the test set.
              </p>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-xs font-medium text-gray-600 mb-1">
              Feature Version
            </label>
            <select
              value={selectedVersionId ?? ''}
              onChange={(e) => setSelectedVersionId(e.target.value ? Number(e.target.value) : null)}
              className="border rounded-md px-3 py-2 text-sm w-full"
            >
              <option value="">Latest (unversioned)</option>
              {versions.map(v => (
                <option key={v.id} value={v.id}>
                  {v.name}{v.description ? ` — ${v.description}` : ''} ({new Date(v.created_at).toLocaleDateString()})
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-400 mt-1">
              Select a versioned snapshot for reproducibility, or use "Latest" for the most recent feature values.
              {versions.length === 0 && ' Create versions in the Feature Builder page.'}
            </p>
          </div>

          <div className="mb-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={autoTune}
                onChange={(e) => setAutoTune(e.target.checked)}
                className="rounded"
              />
              Auto-tune hyperparameters (Optuna)
              {autoTune && (
                <>
                  <input
                    type="number"
                    value={optunTrials}
                    onChange={(e) => setOptunTrials(parseInt(e.target.value) || 50)}
                    min={10}
                    max={1000}
                    className="border rounded px-2 py-1 w-20 text-xs ml-2"
                  />
                  <span className="text-xs text-gray-400">trials</span>
                </>
              )}
            </label>
          </div>

          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-gray-600">
                Features ({selectedFeatures.length} selected)
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedFeatures(features.map(f => f.id))}
                  className="text-xs text-blue-500 hover:underline"
                >
                  Select all
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedFeatures([])}
                  className="text-xs text-gray-500 hover:underline"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto border rounded-md p-2">
              {features.map(f => (
                <label key={f.id} className="flex items-center gap-1 text-xs bg-gray-50 rounded px-2 py-1 cursor-pointer hover:bg-gray-100">
                  <input
                    type="checkbox"
                    checked={selectedFeatures.includes(f.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedFeatures([...selectedFeatures, f.id]);
                      } else {
                        setSelectedFeatures(selectedFeatures.filter(id => id !== f.id));
                      }
                    }}
                    className="rounded"
                  />
                  {f.display_name || f.name}
                </label>
              ))}
            </div>
          </div>

          {/* Summary */}
          {algorithm === 'lambdarank' && (
            <div className="bg-purple-50 border border-purple-200 rounded-md p-3 mb-4 text-sm text-purple-800">
              <strong>LambdaRank</strong> is a learning-to-rank model that sees all dogs in a race simultaneously,
              rather than predicting each dog independently. It learns which dog is most likely to win by comparing
              the entire field. This produces better-calibrated probabilities and more reliable confidence scores.
              The target is always <strong>finish_position</strong> (used internally as relevance labels).
            </div>
          )}

          <div className="bg-blue-50 border border-blue-200 rounded-md p-3 mb-4 text-sm text-blue-800">
            <strong>Training plan:</strong> Train {algorithm === 'lambdarank' ? 'LambdaRank (LightGBM Ranker)' : algorithm.toUpperCase()} to predict <strong>{algorithm === 'lambdarank' ? 'race ranking (win probability derived via softmax)' : target}</strong> using {selectedFeatures.length} features
            {selectedVersionId ? <> from version <strong>{versions.find(v => v.id === selectedVersionId)?.name}</strong></> : ' (latest unversioned)'}.
            Train on all data before <strong>{testAfter}</strong>, test on data after.
            {autoTune && ` Auto-tune with ${optunTrials} Optuna trials.`}
            {' '}Betting P&L will be evaluated by simulating $1 flat bets + Kelly criterion on the test set.
          </div>

          <button
            onClick={handleCreate}
            disabled={creating}
            className="bg-green-600 text-white px-6 py-2 rounded-md text-sm hover:bg-green-700 disabled:opacity-50"
          >
            {creating ? 'Starting...' : 'Start Training'}
          </button>
        </div>
      )}

      {/* Experiments Table */}
      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : experiments.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 text-lg">No experiments yet</p>
          <p className="text-gray-400 text-sm mt-1">Create one above to start training</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
          <table className="w-full text-sm text-left min-w-[700px]">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-3 sm:px-4 py-3">Name</th>
                <th className="px-3 sm:px-4 py-3">Algorithm</th>
                <th className="px-3 sm:px-4 py-3">Target</th>
                <th className="px-3 sm:px-4 py-3">Status</th>
                <th className="px-3 sm:px-4 py-3">Performance</th>
                <th className="px-3 sm:px-4 py-3">Duration</th>
                <th className="px-3 sm:px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link to={`/training/${exp.id}`} className="text-blue-600 hover:underline font-medium">
                      {exp.name}
                    </Link>
                    {exp.split_config && (exp.split_config as any).test_after && (
                      <p className="text-xs text-gray-400">Test after: {(exp.split_config as any).test_after}</p>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {exp.algorithm}
                    {exp.metrics?.optuna_n_trials && (
                      <span className="ml-1.5 text-xs bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded-full">
                        Optuna: {exp.metrics.optuna_n_trials} trials
                      </span>
                    )}
                    {exp.hyperparameters && Object.keys(exp.hyperparameters).length > 0 && (
                      <p className="text-xs text-gray-400 mt-1 truncate max-w-xs" title={Object.entries(exp.hyperparameters).map(([k, v]) => `${k}: ${typeof v === 'number' ? (v % 1 === 0 ? v : Number(v).toPrecision(3)) : v}`).join(', ')}>
                        {Object.entries(exp.hyperparameters).map(([k, v]) =>
                          `${k}: ${typeof v === 'number' ? (v % 1 === 0 ? v : Number(v).toPrecision(3)) : v}`
                        ).join(', ')}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3">{exp.target}</td>
                  <td className="px-4 py-3">
                    {exp.status === 'running' ? (() => {
                      const age = heartbeatAgeSeconds(exp.heartbeat_at);
                      const isStale = age !== null && age > STALE_THRESHOLD_S;
                      return (
                        <div>
                          <span className={`px-2 py-0.5 rounded-full text-xs ${isStale ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>
                            {isStale ? 'stale' : 'running'}
                          </span>
                          {exp.training_stage && (
                            <p className="text-xs text-gray-500 mt-1">{formatStage(exp.training_stage)}</p>
                          )}
                          {age !== null && (
                            <p className={`text-xs mt-0.5 ${isStale ? 'text-red-500 font-medium' : 'text-gray-400'}`}>
                              {isStale ? `No heartbeat for ${formatHeartbeatAge(age)}` : `Heartbeat ${formatHeartbeatAge(age)}`}
                            </p>
                          )}
                          {age === null && (
                            <p className="text-xs mt-0.5 text-red-500 font-medium">No heartbeat</p>
                          )}
                        </div>
                      );
                    })() : (
                      <div>
                        <span className={`px-2 py-0.5 rounded-full text-xs ${statusColor(exp.status)}`}>
                          {exp.status}
                        </span>
                        {exp.status === 'failed' && exp.error_message && (
                          <p className="text-xs text-red-600 mt-1 max-w-xs truncate" title={exp.error_message}>
                            {exp.error_message.split('\n')[0]}
                          </p>
                        )}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs">{primaryMetric(exp)}</td>
                  <td className="px-4 py-3 text-xs text-gray-500">
                    {exp.training_duration_s ? `${exp.training_duration_s.toFixed(1)}s` : '-'}
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => handleDelete(exp.id)}
                      className="text-red-500 hover:text-red-700 text-xs"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
