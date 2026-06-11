import { useEffect, useState, useCallback, useMemo } from 'react';
import Editor from '@monaco-editor/react';
import api, { errorMessage } from '../api/client';
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

  // Coverage tracking
  const [coverage, setCoverage] = useState<{
    feature_id: number; name: string; display_name: string | null;
    computed_count: number; incomplete_count: number; total_entries: number; coverage_pct: number;
  }[]>([]);
  const [showCoverage, setShowCoverage] = useState(false);

  // Auto-injected feature groups (ELO, speed figures, H2H, etc. — computed
  // on-the-fly during training/prediction, not materialized)
  type AutoGroup = {
    key: string;
    title: string;
    toggle_flag: string;
    description: string;
    features: string[];
    base_columns?: string[];
  };
  const [autoGroups, setAutoGroups] = useState<AutoGroup[] | null>(null);
  const [autoTotal, setAutoTotal] = useState(0);
  const [showAutoInjected, setShowAutoInjected] = useState(false);
  const [expandedAuto, setExpandedAuto] = useState<Record<string, boolean>>({});

  // Versioning
  const [versions, setVersions] = useState<{
    id: number; name: string; description: string | null;
    created_at: string; feature_count: number;
    coverage_snapshot: { recommendation?: string } | null;
  }[]>([]);
  const [showVersions, setShowVersions] = useState(false);
  const [newVersionName, setNewVersionName] = useState('');
  const [newVersionDesc, setNewVersionDesc] = useState('');
  const [creatingVersion, setCreatingVersion] = useState(false);
  const [selectedVersionId, setSelectedVersionId] = useState<number | null>(null);

  // Visual builder state
  const [visualConfig, setVisualConfig] = useState<VisualConfig>({
    metric: 'finish_time',
    aggregation: 'mean',
    window: { type: 'last_n', n: 5 },
    filters: { same_track: false, same_distance: false, same_grade: false, same_trap: false },
  });
  // Names are derived purely from the visual config — computed during render
  // (memoised) rather than synced into state via an effect.
  const { visualName, visualDisplayName } = useMemo(() => {
    const { metric, aggregation, window: w, filters } = visualConfig;
    const parts = [aggregation, metric];
    if (w.type === 'last_n') parts.push(`last${w.n}`);
    else if (w.type === 'days') parts.push(`${w.n}d`);
    if (filters.same_track) parts.push('track');
    if (filters.same_distance) parts.push('dist');
    if (filters.same_grade) parts.push('grade');
    if (filters.same_trap) parts.push('trap');

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

    return { visualName: parts.join('_'), visualDisplayName: displayParts.join(' ') };
  }, [visualConfig]);

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

  const handlePreview = async () => {
    setPreviewing(true);
    setPreviewResult(null);
    try {
      const body = activeTab === 'visual'
        ? { feature_type: 'visual', config_json: visualConfig, dog_id: parseInt(previewDogId) }
        : { feature_type: 'code', code, dog_id: parseInt(previewDogId) };
      const res = await api.post('/features/preview', body);
      setPreviewResult(res.data);
    } catch (err) {
      setPreviewResult({ value: null, error: errorMessage(err, 'Preview failed') });
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
    } catch (err) {
      alert(errorMessage(err, 'Failed to save feature'));
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

  const fetchCoverage = useCallback((versionId?: number | null) => {
    const vid = versionId !== undefined ? versionId : selectedVersionId;
    const params: Record<string, number> = {};
    if (vid) params.version_id = vid;
    api.get('/features/coverage', { params }).then(res => {
      setCoverage(res.data);
    }).catch(() => {});
  }, [selectedVersionId]);

  const fetchVersions = useCallback(() => {
    api.get('/features/versions').then(res => {
      setVersions(res.data);
    }).catch(() => {});
  }, []);

  const fetchAutoInjected = useCallback(() => {
    if (autoGroups !== null) return;  // cache — registries are static
    api.get('/features/auto-injected').then(res => {
      setAutoGroups(res.data.groups);
      setAutoTotal(res.data.total);
    }).catch(() => {});
  }, [autoGroups]);

  const handleCreateVersion = async () => {
    if (!newVersionName.trim()) return;
    setCreatingVersion(true);
    try {
      const res = await api.post('/features/versions', {
        name: newVersionName.trim(),
        description: newVersionDesc.trim() || null,
      });
      setNewVersionName('');
      setNewVersionDesc('');
      setSelectedVersionId(res.data.id);
      fetchVersions();
    } catch (err) {
      alert(errorMessage(err, 'Failed to create version'));
    }
    setCreatingVersion(false);
  };

  const handleDeleteVersion = async (id: number) => {
    if (!confirm('Delete this version and all its computed features?')) return;
    await api.delete(`/features/versions/${id}`);
    if (selectedVersionId === id) setSelectedVersionId(null);
    fetchVersions();
  };

  const handleMaterialize = async () => {
    setMaterializing(true);
    setShowCoverage(true);
    try {
      const body: { force: boolean; version_id?: number } = { force: false };
      if (selectedVersionId) body.version_id = selectedVersionId;
      await api.post('/features/materialize', body);
    } catch {
      alert('Failed to start materialization');
    }
    setMaterializing(false);
  };

  // Clear old polling interval when version changes, start new one if coverage is visible
  useEffect(() => {
    const w = window as Window & {
      __materializeInterval?: ReturnType<typeof setInterval> | null;
    };
    if (w.__materializeInterval) {
      clearInterval(w.__materializeInterval);
      w.__materializeInterval = null;
    }
    if (showCoverage) {
      fetchCoverage(selectedVersionId);
      const params: Record<string, number> = {};
      if (selectedVersionId) params.version_id = selectedVersionId;
      const interval = setInterval(() => {
        api.get('/features/coverage', { params }).then(res => setCoverage(res.data));
      }, 5000);
      w.__materializeInterval = interval;
      return () => clearInterval(interval);
    }
  }, [showCoverage, selectedVersionId, fetchCoverage]);

  // Fetch versions on mount
  useEffect(() => { fetchVersions(); }, [fetchVersions]);

  return (
    <div>
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 sm:mb-6">
        <h1 className="text-xl sm:text-2xl font-bold">Feature Builder</h1>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => { setShowVersions(!showVersions); }}
            className="bg-gray-100 text-gray-700 px-3 sm:px-4 py-2 rounded-md text-sm hover:bg-gray-200"
          >
            {showVersions ? 'Hide Versions' : 'Versions'}
          </button>
          <button
            onClick={() => { setShowCoverage(!showCoverage); if (!showCoverage) fetchCoverage(); }}
            className="bg-gray-100 text-gray-700 px-3 sm:px-4 py-2 rounded-md text-sm hover:bg-gray-200"
          >
            {showCoverage ? 'Hide Progress' : 'Show Progress'}
          </button>
          <button
            onClick={() => { setShowAutoInjected(!showAutoInjected); if (!showAutoInjected) fetchAutoInjected(); }}
            className="bg-gray-100 text-gray-700 px-3 sm:px-4 py-2 rounded-md text-sm hover:bg-gray-200"
          >
            {showAutoInjected ? 'Hide Auto-Injected' : 'Auto-Injected Features'}
          </button>
          <button
            onClick={handleMaterialize}
            disabled={materializing}
            className="bg-green-600 text-white px-3 sm:px-4 py-2 rounded-md text-sm hover:bg-green-700 disabled:opacity-50"
          >
            {materializing ? 'Starting...' : selectedVersionId ? `Materialize into "${versions.find(v => v.id === selectedVersionId)?.name || selectedVersionId}"` : 'Materialize All'}
          </button>
        </div>
      </div>

      {/* Version Management */}
      {showVersions && (
        <div className="bg-white rounded-lg shadow p-5 mb-6">
          <h2 className="font-semibold mb-3">Feature Versions</h2>
          <p className="text-xs text-gray-500 mb-4">
            Create named snapshots of your features. Each version captures the current scrape state
            so you can compare models trained on different data.
          </p>

          {/* Create new version */}
          <div className="flex flex-col sm:flex-row gap-2 mb-4">
            <input
              type="text"
              value={newVersionName}
              onChange={(e) => setNewVersionName(e.target.value)}
              placeholder="Version name (e.g. v1-full-scrape)"
              className="border rounded-md px-3 py-2 text-sm flex-1"
            />
            <input
              type="text"
              value={newVersionDesc}
              onChange={(e) => setNewVersionDesc(e.target.value)}
              placeholder="Description (optional)"
              className="border rounded-md px-3 py-2 text-sm flex-1"
            />
            <button
              onClick={handleCreateVersion}
              disabled={creatingVersion || !newVersionName.trim()}
              className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50 whitespace-nowrap"
            >
              {creatingVersion ? 'Creating...' : 'Create Version'}
            </button>
          </div>

          {/* Version list */}
          <div className="space-y-2">
            {/* Unversioned option */}
            <div
              onClick={() => setSelectedVersionId(null)}
              className={`flex items-center justify-between p-3 rounded-md border cursor-pointer transition-colors ${
                selectedVersionId === null ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'
              }`}
            >
              <div>
                <span className="text-sm font-medium">Unversioned (default)</span>
                <p className="text-xs text-gray-500">Features are upserted in place</p>
              </div>
              {selectedVersionId === null && (
                <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">Selected</span>
              )}
            </div>

            {versions.map((v) => (
              <div
                key={v.id}
                onClick={() => setSelectedVersionId(v.id)}
                className={`flex items-center justify-between p-3 rounded-md border cursor-pointer transition-colors ${
                  selectedVersionId === v.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{v.name}</span>
                    {v.coverage_snapshot?.recommendation && (
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        v.coverage_snapshot.recommendation === 'safe'
                          ? 'bg-green-100 text-green-700'
                          : v.coverage_snapshot.recommendation === 'warning'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {v.coverage_snapshot.recommendation}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 truncate">
                    {v.description || 'No description'}
                    {' \u00b7 '}{v.feature_count.toLocaleString()} features
                    {' \u00b7 '}{new Date(v.created_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="flex items-center gap-2 ml-2">
                  {selectedVersionId === v.id && (
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">Selected</span>
                  )}
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteVersion(v.id); }}
                    className="text-red-400 hover:text-red-600 text-xs px-1"
                    title="Delete version"
                  >
                    x
                  </button>
                </div>
              </div>
            ))}

            {versions.length === 0 && (
              <p className="text-xs text-gray-400 text-center py-2">No versions created yet</p>
            )}
          </div>
        </div>
      )}

      {/* Auto-injected feature inventory */}
      {showAutoInjected && (
        <div className="bg-white rounded-lg shadow p-5 mb-6">
          <div className="flex items-start justify-between gap-3 mb-3">
            <div>
              <h2 className="font-semibold">Auto-Injected Features</h2>
              <p className="text-xs text-gray-500 mt-1 max-w-3xl">
                These features are <strong>not stored</strong> in the
                computed_features table — they are computed on the fly during
                training and prediction. Some need the full race field or a
                chronological walk across every race (ELO, pace-shape,
                head-to-head), so they can&apos;t fit the per-entry
                materialization model. They&apos;re listed here purely so you
                can see what the model actually trains on. Each group is
                enabled by default and can be toggled via the corresponding
                flag on the Training Lab form.
              </p>
            </div>
            {autoGroups && (
              <div className="text-sm text-gray-600 whitespace-nowrap">
                <strong>{autoTotal}</strong> features across{' '}
                <strong>{autoGroups.length}</strong> groups
              </div>
            )}
          </div>

          {!autoGroups && (
            <div className="text-sm text-gray-500">Loading…</div>
          )}

          {autoGroups && (
            <div className="space-y-3">
              {autoGroups.map((g) => {
                const isOpen = !!expandedAuto[g.key];
                return (
                  <div key={g.key} className="border rounded-md">
                    <button
                      type="button"
                      onClick={() => setExpandedAuto((s) => ({ ...s, [g.key]: !s[g.key] }))}
                      className="w-full flex items-center justify-between text-left px-3 py-2 hover:bg-gray-50"
                    >
                      <div>
                        <div className="font-medium text-sm">
                          {g.title}
                          <span className="text-xs text-gray-400 font-normal ml-2">
                            {g.features.length} features
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 mt-0.5">
                          Toggle: <code className="text-[11px]">{g.toggle_flag}</code>
                        </div>
                      </div>
                      <span className="text-gray-400 text-xs">{isOpen ? '▾' : '▸'}</span>
                    </button>
                    {isOpen && (
                      <div className="border-t px-3 py-3 bg-gray-50">
                        <p className="text-xs text-gray-600 mb-2">{g.description}</p>
                        {g.key === 'race_relative' && g.base_columns && (
                          <p className="text-xs text-gray-500 mb-2">
                            Base columns ({g.base_columns.length}): each gets
                            5 variants (<code>__vs_field</code>,{' '}
                            <code>__rank</code>, <code>__z_in_field</code>,{' '}
                            <code>__gap_to_best</code>,{' '}
                            <code>__is_field_best</code>).
                          </p>
                        )}
                        <div className="flex flex-wrap gap-1.5">
                          {(g.key === 'race_relative' && g.base_columns
                            ? g.base_columns
                            : g.features
                          ).map((name) => (
                            <code
                              key={name}
                              className="text-[11px] bg-white border border-gray-200 rounded px-1.5 py-0.5 text-gray-700"
                            >
                              {name}
                            </code>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Materialization Progress */}
      {showCoverage && (
        <div className="bg-white rounded-lg shadow p-5 mb-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Materialization Progress</h2>
            <button onClick={() => fetchCoverage()} className="text-xs text-blue-500 hover:underline">
              Refresh
            </button>
          </div>
          {coverage.length === 0 ? (
            <p className="text-gray-400 text-sm">Loading coverage data...</p>
          ) : (
            <div className="space-y-2">
              {coverage.map((c) => {
                const pct = c.coverage_pct;
                const isComplete = pct >= 99;
                return (
                  <div key={c.feature_id}>
                    <div className="flex items-center justify-between text-xs mb-0.5">
                      <span className="text-gray-600 truncate max-w-[250px]">{c.display_name || c.name}</span>
                      <span className={`font-mono ${isComplete ? 'text-green-600' : 'text-gray-500'}`}>
                        {c.computed_count.toLocaleString()} / {c.total_entries.toLocaleString()} ({pct}%)
                        {c.incomplete_count > 0 && (
                          <span className="text-yellow-600 ml-1" title="Features computed with potentially incomplete data">
                            ({c.incomplete_count.toLocaleString()} incomplete)
                          </span>
                        )}
                      </span>
                    </div>
                    <div className="w-full bg-gray-100 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${isComplete ? 'bg-green-500' : pct > 0 ? 'bg-blue-500' : 'bg-gray-200'}`}
                        style={{ width: `${Math.min(pct, 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
              <div className="pt-2 border-t mt-3 text-sm text-gray-500">
                {(() => {
                  const total = coverage.reduce((s, c) => s + c.total_entries, 0);
                  const done = coverage.reduce((s, c) => s + c.computed_count, 0);
                  const overallPct = total > 0 ? (done / total * 100).toFixed(1) : '0';
                  return `Overall: ${done.toLocaleString()} / ${total.toLocaleString()} computations (${overallPct}%)`;
                })()}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="flex flex-col md:flex-row gap-4">
        {/* Feature list sidebar */}
        <div className="w-full md:w-72 md:shrink-0">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold text-sm mb-3">
              Features ({features.length})
            </h2>
            {loading ? (
              <p className="text-gray-500 text-sm">Loading...</p>
            ) : features.length === 0 ? (
              <p className="text-gray-400 text-sm">No features defined yet</p>
            ) : (
              <ul className="space-y-2 max-h-[300px] md:max-h-[600px] overflow-y-auto">
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
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
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
              <div className="flex flex-wrap items-end gap-3 mt-4 pt-4 border-t">
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
