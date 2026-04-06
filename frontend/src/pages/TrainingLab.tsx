import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import type { Experiment, FeatureDefinition } from '../types/models';

const ALGORITHMS = [
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
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [creating, setCreating] = useState(false);

  // Form state
  const [name, setName] = useState('');
  const [algorithm, setAlgorithm] = useState('xgboost');
  const [target, setTarget] = useState('win_prob');
  const [selectedFeatures, setSelectedFeatures] = useState<number[]>([]);
  const [autoTune, setAutoTune] = useState(false);
  const [optunTrials, setOptunTrials] = useState(50);

  const fetchData = () => {
    Promise.all([
      api.get<Experiment[]>('/training/experiments'),
      api.get<FeatureDefinition[]>('/features/?enabled_only=true'),
    ]).then(([expRes, featRes]) => {
      setExperiments(expRes.data);
      setFeatures(featRes.data);
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
      await api.post('/training/experiments', {
        name,
        algorithm,
        target,
        hyperparameters: {},
        feature_set: selectedFeatures,
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

  const primaryMetric = (exp: Experiment) => {
    if (!exp.metrics) return '-';
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
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Training Lab</h1>
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
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. XGBoost Win Prob v1"
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
              <label className="flex items-center gap-2 text-sm mt-6">
                <input
                  type="checkbox"
                  checked={autoTune}
                  onChange={(e) => setAutoTune(e.target.checked)}
                  className="rounded"
                />
                Auto-tune (Optuna)
                {autoTune && (
                  <input
                    type="number"
                    value={optunTrials}
                    onChange={(e) => setOptunTrials(parseInt(e.target.value) || 50)}
                    min={10}
                    max={200}
                    className="border rounded px-2 py-1 w-20 text-xs ml-2"
                  />
                )}
                {autoTune && <span className="text-xs text-gray-400">trials</span>}
              </label>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-xs font-medium text-gray-600 mb-2">
              Features ({selectedFeatures.length} selected)
            </label>
            <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
              {features.map(f => (
                <label key={f.id} className="flex items-center gap-1 text-xs bg-gray-50 rounded px-2 py-1 cursor-pointer">
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
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Algorithm</th>
                <th className="px-4 py-3">Target</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Key Metric</th>
                <th className="px-4 py-3">Duration</th>
                <th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link to={`/training/${exp.id}`} className="text-blue-600 hover:underline font-medium">
                      {exp.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3">{exp.algorithm}</td>
                  <td className="px-4 py-3">{exp.target}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs ${statusColor(exp.status)}`}>
                      {exp.status}
                    </span>
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
