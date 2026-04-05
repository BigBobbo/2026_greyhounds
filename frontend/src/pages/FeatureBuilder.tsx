import { useEffect, useState, useCallback } from 'react';
import Editor from '@monaco-editor/react';
import api from '../api/client';
import type { FeatureDefinition } from '../types/models';

const METRICS = [
  { value: 'finish_time', label: 'Finish Time' },
  { value: 'finish_position', label: 'Finish Position' },
  { value: 'weight_kg', label: 'Weight (kg)' },
  { value: 'sp_decimal', label: 'Starting Price (decimal)' },
  { value: 'beaten_distance', label: 'Beaten Distance' },
  { value: 'sectional_time', label: 'Sectional Time' },
];

const AGGREGATIONS = [
  { value: 'mean', label: 'Mean' },
  { value: 'median', label: 'Median' },
  { value: 'min', label: 'Min' },
  { value: 'max', label: 'Max' },
  { value: 'stdev', label: 'Std Dev' },
  { value: 'count', label: 'Count' },
  { value: 'win_rate', label: 'Win Rate' },
  { value: 'place_rate', label: 'Place Rate (top 3)' },
  { value: 'trend', label: 'Trend (slope)' },
];

const WINDOW_TYPES = [
  { value: 'last_n', label: 'Last N races' },
  { value: 'days', label: 'Last N days' },
  { value: 'all', label: 'All history' },
];

const DEFAULT_CODE = `def compute(dog_history, race_context):
    """
    dog_history: pd.DataFrame with columns:
        trap, finish_position, finish_time, sectional_time,
        beaten_distance, weight_kg, sp_decimal, race_date,
        track_id, distance_m, grade, race_type, going,
        num_runners, track_name, track_code
    race_context: dict with keys:
        trap, dog_id, track_id, distance_m, grade, race_date,
        race_type, track_code
    Returns: float or None
    """
    recent = dog_history.tail(5)
    if len(recent) == 0:
        return None
    return float(recent['finish_time'].mean())
`;

interface VisualConfig {
  metric: string;
  aggregation: string;
  window: { type: string; n: number };
  filters: {
    same_track: boolean;
    same_distance: boolean;
    same_grade: boolean;
    same_trap: boolean;
  };
}

export default function FeatureBuilder() {
  const [features, setFeatures] = useState<FeatureDefinition[]>([]);
  const [activeTab, setActiveTab] = useState<'visual' | 'code'>('visual');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [materializing, setMaterializing] = useState(false);

  // Visual builder state
  const [visualConfig, setVisualConfig] = useState<VisualConfig>({
    metric: 'finish_time',
    aggregation: 'mean',
    window: { type: 'last_n', n: 5 },
    filters: { same_track: false, same_distance: false, same_grade: false, same_trap: false },
  });
  const [visualName, setVisualName] = useState('');
  const [visualDisplayName, setVisualDisplayName] = useState('');

  // Code editor state
  const [code, setCode] = useState(DEFAULT_CODE);
  const [codeName, setCodeName] = useState('');
  const [codeDisplayName, setCodeDisplayName] = useState('');

  // Preview state
  const [previewDogId, setPreviewDogId] = useState('1');
  const [previewResult, setPreviewResult] = useState<{ value: number | null; error: string | null } | null>(null);
  const [previewing, setPreviewing] = useState(false);

  const fetchFeatures = useCallback(() => {
    api.get<FeatureDefinition[]>('/features/').then((res) => {
      setFeatures(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => { fetchFeatures(); }, [fetchFeatures]);

  // Auto-generate name from visual config
  useEffect(() => {
    const { metric, aggregation, window: w, filters } = visualConfig;
    const parts = [aggregation, metric];
    if (w.type === 'last_n') parts.push(`last${w.n}`);
    else if (w.type === 'days') parts.push(`${w.n}d`);
    if (filters.same_track) parts.push('track');
    if (filters.same_distance) parts.push('dist');
    if (filters.same_grade) parts.push('grade');
    if (filters.same_trap) parts.push('trap');
    setVisualName(parts.join('_'));

    const displayParts = [
      AGGREGATIONS.find(a => a.value === aggregation)?.label || aggregation,
      METRICS.find(m => m.value === metric)?.label || metric,
    ];
    if (w.type === 'last_n') displayParts.push(`(last ${w.n})`);
    else if (w.type === 'days') displayParts.push(`(${w.n} days)`);
    const filterLabels = [];
    if (filters.same_track) filterLabels.push('same track');
    if (filters.same_distance) filterLabels.push('same distance');
    if (filters.same_grade) filterLabels.push('same grade');
    if (filters.same_trap) filterLabels.push('same trap');
    if (filterLabels.length) displayParts.push(`[${filterLabels.join(', ')}]`);
    setVisualDisplayName(displayParts.join(' '));
  }, [visualConfig]);

  const handlePreview = async () => {
    setPreviewing(true);
    setPreviewResult(null);
    try {
      const body = activeTab === 'visual'
        ? { feature_type: 'visual', config_json: visualConfig, dog_id: parseInt(previewDogId) }
        : { feature_type: 'code', code, dog_id: parseInt(previewDogId) };
      const res = await api.post('/features/preview', body);
      setPreviewResult(res.data);
    } catch (err: any) {
      setPreviewResult({ value: null, error: err.response?.data?.detail || 'Preview failed' });
    }
    setPreviewing(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const body = activeTab === 'visual'
        ? {
            name: visualName,
            display_name: visualDisplayName,
            feature_type: 'visual',
            config_json: visualConfig,
            input_columns: [visualConfig.metric],
          }
        : {
            name: codeName || 'custom_feature_' + Date.now(),
            display_name: codeDisplayName || codeName,
            feature_type: 'code',
            code,
          };
      await api.post('/features/', body);
      fetchFeatures();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to save feature');
    }
    setSaving(false);
  };

  const handleToggle = async (id: number, enabled: boolean) => {
    await api.patch(`/features/${id}`, { enabled: !enabled });
    fetchFeatures();
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this feature?')) return;
    await api.delete(`/features/${id}`);
    fetchFeatures();
  };

  const handleMaterialize = async () => {
    setMaterializing(true);
    try {
      await api.post('/features/materialize', { force: false });
      alert('Materialization started in background');
    } catch {
      alert('Failed to start materialization');
    }
    setMaterializing(false);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Feature Builder</h1>
        <button
          onClick={handleMaterialize}
          disabled={materializing}
          className="bg-green-600 text-white px-4 py-2 rounded-md text-sm hover:bg-green-700 disabled:opacity-50"
        >
          {materializing ? 'Materializing...' : 'Materialize All'}
        </button>
      </div>

      <div className="flex gap-4">
        {/* Feature list sidebar */}
        <div className="w-72 shrink-0">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold text-sm mb-3">
              Features ({features.length})
            </h2>
            {loading ? (
              <p className="text-gray-500 text-sm">Loading...</p>
            ) : features.length === 0 ? (
              <p className="text-gray-400 text-sm">No features defined yet</p>
            ) : (
              <ul className="space-y-2 max-h-[600px] overflow-y-auto">
                {features.map((f) => (
                  <li key={f.id} className="border rounded-md p-2 text-sm">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate">{f.display_name || f.name}</p>
                        <p className="text-xs text-gray-400">{f.feature_type}</p>
                      </div>
                      <div className="flex items-center gap-1 ml-2">
                        <button
                          onClick={() => handleToggle(f.id, f.enabled)}
                          className={`w-8 h-4 rounded-full transition-colors ${f.enabled ? 'bg-green-500' : 'bg-gray-300'}`}
                          title={f.enabled ? 'Disable' : 'Enable'}
                        >
                          <span className={`block w-3 h-3 rounded-full bg-white transition-transform ${f.enabled ? 'translate-x-4' : 'translate-x-0.5'}`} />
                        </button>
                        <button
                          onClick={() => handleDelete(f.id)}
                          className="text-red-400 hover:text-red-600 text-xs px-1"
                          title="Delete"
                        >
                          x
                        </button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Builder area */}
        <div className="flex-1">
          <div className="bg-white rounded-lg shadow">
            {/* Tabs */}
            <div className="flex border-b">
              <button
                onClick={() => setActiveTab('visual')}
                className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'visual' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Visual Builder
              </button>
              <button
                onClick={() => setActiveTab('code')}
                className={`px-5 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'code' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                Code Editor
              </button>
            </div>

            <div className="p-5">
              {activeTab === 'visual' ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Metric</label>
                      <select
                        value={visualConfig.metric}
                        onChange={(e) => setVisualConfig({ ...visualConfig, metric: e.target.value })}
                        className="border rounded-md px-3 py-2 text-sm w-full"
                      >
                        {METRICS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Aggregation</label>
                      <select
                        value={visualConfig.aggregation}
                        onChange={(e) => setVisualConfig({ ...visualConfig, aggregation: e.target.value })}
                        className="border rounded-md px-3 py-2 text-sm w-full"
                      >
                        {AGGREGATIONS.map(a => <option key={a.value} value={a.value}>{a.label}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Window Type</label>
                      <select
                        value={visualConfig.window.type}
                        onChange={(e) => setVisualConfig({
                          ...visualConfig,
                          window: { ...visualConfig.window, type: e.target.value },
                        })}
                        className="border rounded-md px-3 py-2 text-sm w-full"
                      >
                        {WINDOW_TYPES.map(w => <option key={w.value} value={w.value}>{w.label}</option>)}
                      </select>
                    </div>
                    {visualConfig.window.type !== 'all' && (
                      <div>
                        <label className="block text-xs font-medium text-gray-600 mb-1">
                          {visualConfig.window.type === 'last_n' ? 'Number of races' : 'Number of days'}
                        </label>
                        <input
                          type="number"
                          value={visualConfig.window.n}
                          onChange={(e) => setVisualConfig({
                            ...visualConfig,
                            window: { ...visualConfig.window, n: parseInt(e.target.value) || 5 },
                          })}
                          min={1}
                          max={visualConfig.window.type === 'days' ? 365 : 50}
                          className="border rounded-md px-3 py-2 text-sm w-full"
                        />
                      </div>
                    )}
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-2">Filters</label>
                    <div className="flex flex-wrap gap-3">
                      {[
                        { key: 'same_track' as const, label: 'Same Track' },
                        { key: 'same_distance' as const, label: 'Same Distance' },
                        { key: 'same_grade' as const, label: 'Same Grade' },
                        { key: 'same_trap' as const, label: 'Same Trap' },
                      ].map(({ key, label }) => (
                        <label key={key} className="flex items-center gap-1.5 text-sm cursor-pointer">
                          <input
                            type="checkbox"
                            checked={visualConfig.filters[key]}
                            onChange={(e) => setVisualConfig({
                              ...visualConfig,
                              filters: { ...visualConfig.filters, [key]: e.target.checked },
                            })}
                            className="rounded"
                          />
                          {label}
                        </label>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-md p-3 text-sm">
                    <p className="text-gray-500">Generated name:</p>
                    <p className="font-mono text-xs mt-1">{visualName}</p>
                    <p className="text-gray-500 mt-1">Display:</p>
                    <p className="text-xs mt-0.5">{visualDisplayName}</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Feature Name</label>
                      <input
                        type="text"
                        value={codeName}
                        onChange={(e) => setCodeName(e.target.value)}
                        placeholder="my_custom_feature"
                        className="border rounded-md px-3 py-2 text-sm w-full"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">Display Name</label>
                      <input
                        type="text"
                        value={codeDisplayName}
                        onChange={(e) => setCodeDisplayName(e.target.value)}
                        placeholder="My Custom Feature"
                        className="border rounded-md px-3 py-2 text-sm w-full"
                      />
                    </div>
                  </div>
                  <div className="border rounded-md overflow-hidden">
                    <Editor
                      height="350px"
                      defaultLanguage="python"
                      value={code}
                      onChange={(v) => setCode(v || '')}
                      theme="vs-dark"
                      options={{
                        minimap: { enabled: false },
                        fontSize: 13,
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                        wordWrap: 'on',
                        tabSize: 4,
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Preview + Save row */}
              <div className="flex items-end gap-3 mt-4 pt-4 border-t">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Dog ID (for preview)</label>
                  <input
                    type="number"
                    value={previewDogId}
                    onChange={(e) => setPreviewDogId(e.target.value)}
                    min={1}
                    className="border rounded-md px-3 py-2 text-sm w-24"
                  />
                </div>
                <button
                  onClick={handlePreview}
                  disabled={previewing}
                  className="bg-gray-600 text-white px-4 py-2 rounded-md text-sm hover:bg-gray-700 disabled:opacity-50"
                >
                  {previewing ? 'Computing...' : 'Preview'}
                </button>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
                >
                  {saving ? 'Saving...' : 'Save Feature'}
                </button>

                {previewResult && (
                  <div className={`ml-3 text-sm ${previewResult.error ? 'text-red-600' : 'text-green-700'}`}>
                    {previewResult.error
                      ? `Error: ${previewResult.error}`
                      : `Value: ${previewResult.value !== null ? previewResult.value.toFixed(4) : 'null'}`}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
