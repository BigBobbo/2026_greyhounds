import { useEffect, useState, useCallback } from 'react';
import api from '../api/client';

interface Experiment {
  id: number;
  name: string;
  algorithm: string;
  target: string;
  status: string;
  metrics: Record<string, number> | null;
}

interface ScheduleRun {
  id: number;
  model_schedule_id: number;
  run_date: string;
  status: string;
  trigger: string;
  races_predicted: number;
  races_skipped: number;
  predictions_written: number;
  error_message: string | null;
  started_at: string;
  finished_at: string | null;
}

interface ModelSchedule {
  id: number;
  experiment_id: number;
  experiment_name: string | null;
  experiment_status: string | null;
  enabled: boolean;
  is_main: boolean;
  cron_hour: number;
  cron_minute: number;
  timezone: string;
  scrape_upcoming: boolean;
  predict_days_ahead: number;
  created_at: string;
  updated_at: string;
  last_run: ScheduleRun | null;
}

interface PerformanceBucket {
  bucket: number;
  mean_predicted: number | null;
  empirical: number | null;
  n: number;
}

interface Performance {
  schedule_id: number;
  experiment_id: number;
  window_days: number;
  races_evaluated: number;
  top1_accuracy: number | null;
  top3_hit_rate: number | null;
  mean_log_loss: number | null;
  calibration: PerformanceBucket[];
}

function pad(n: number) {
  return n.toString().padStart(2, '0');
}

function statusBadge(status: string): string {
  const map: Record<string, string> = {
    success: 'bg-green-100 text-green-700',
    partial: 'bg-yellow-100 text-yellow-700',
    failed: 'bg-red-100 text-red-700',
    running: 'bg-blue-100 text-blue-700',
  };
  return map[status] || 'bg-gray-100 text-gray-700';
}

export default function Schedule() {
  const [schedules, setSchedules] = useState<ModelSchedule[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [perfById, setPerfById] = useState<Record<number, Performance>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // New-schedule form state
  const [newExpId, setNewExpId] = useState<number | ''>('');
  const [newHour, setNewHour] = useState(8);
  const [newMinute, setNewMinute] = useState(30);
  const [newAsMain, setNewAsMain] = useState(false);
  const [creating, setCreating] = useState(false);

  // Per-row pending action
  const [busyRow, setBusyRow] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [schedRes, expRes] = await Promise.all([
        api.get<ModelSchedule[]>('/schedule'),
        api.get<Experiment[]>('/training/experiments?limit=100'),
      ]);
      setSchedules(schedRes.data);
      setExperiments(expRes.data.filter((e) => e.status === 'completed'));

      // Pull performance for each schedule (small N).
      const perfEntries = await Promise.all(
        schedRes.data.map((s) =>
          api
            .get<Performance>(`/schedule/${s.id}/performance?days=30`)
            .then((r) => [s.id, r.data] as const)
            .catch(() => null),
        ),
      );
      const perfMap: Record<number, Performance> = {};
      for (const entry of perfEntries) {
        if (entry) perfMap[entry[0]] = entry[1];
      }
      setPerfById(perfMap);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to load schedules';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleCreate = async () => {
    if (newExpId === '') return;
    setCreating(true);
    try {
      await api.post('/schedule', {
        experiment_id: newExpId,
        cron_hour: newHour,
        cron_minute: newMinute,
        is_main: newAsMain,
      });
      setNewExpId('');
      setNewAsMain(false);
      await refresh();
    } catch {
      // error toast shown by the API interceptor
    } finally {
      setCreating(false);
    }
  };

  const patchSchedule = async (id: number, patch: Partial<ModelSchedule>) => {
    setBusyRow(id);
    try {
      await api.patch(`/schedule/${id}`, patch);
      await refresh();
    } catch {
      // error toast shown by the API interceptor
    } finally {
      setBusyRow(null);
    }
  };

  const handleRun = async (id: number) => {
    setBusyRow(id);
    try {
      await api.post(`/schedule/${id}/run`);
      // The job runs in a background thread on the server. Wait briefly,
      // then refresh so the audit log row appears with its eventual status.
      setTimeout(refresh, 1500);
    } catch {
      // error toast shown by the API interceptor
    } finally {
      setBusyRow(null);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this schedule? Predictions already written will be kept.')) return;
    setBusyRow(id);
    try {
      await api.delete(`/schedule/${id}`);
      await refresh();
    } catch {
      // error toast shown by the API interceptor
    } finally {
      setBusyRow(null);
    }
  };

  // Available experiments for the picker (exclude already-scheduled).
  const scheduledExpIds = new Set(schedules.map((s) => s.experiment_id));
  const availableExps = experiments.filter((e) => !scheduledExpIds.has(e.id));

  return (
    <div>
      <div className="flex items-start justify-between mb-4 sm:mb-6 gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold">Schedule</h1>
          <p className="text-sm text-gray-500 mt-1">
            Daily prediction tracking. The "main" model runs by default; tracked models log
            predictions for comparison without affecting your bankroll.
          </p>
        </div>
        <button
          onClick={refresh}
          className="px-3 py-1.5 bg-gray-200 rounded text-sm hover:bg-gray-300"
        >
          Refresh
        </button>
      </div>

      {/* Add schedule */}
      <div className="bg-white rounded-lg shadow p-4 sm:p-5 mb-5">
        <h2 className="font-semibold mb-3">Add a model to the daily schedule</h2>
        {availableExps.length === 0 ? (
          <p className="text-sm text-gray-500">
            No completed experiments available to schedule. Train one in the Training Lab first.
          </p>
        ) : (
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Experiment</label>
              <select
                className="border rounded px-2 py-1.5 text-sm min-w-[14rem]"
                value={newExpId}
                onChange={(e) =>
                  setNewExpId(e.target.value === '' ? '' : Number(e.target.value))
                }
              >
                <option value="">— pick a model —</option>
                {availableExps.map((e) => (
                  <option key={e.id} value={e.id}>
                    #{e.id} {e.name} ({e.algorithm})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Run time (local)</label>
              <div className="flex items-center gap-1">
                <input
                  type="number"
                  min={0}
                  max={23}
                  className="w-16 border rounded px-2 py-1.5 text-sm"
                  value={newHour}
                  onChange={(e) => setNewHour(Number(e.target.value))}
                />
                <span>:</span>
                <input
                  type="number"
                  min={0}
                  max={59}
                  className="w-16 border rounded px-2 py-1.5 text-sm"
                  value={newMinute}
                  onChange={(e) => setNewMinute(Number(e.target.value))}
                />
                <span className="text-xs text-gray-500 ml-1">Europe/Dublin</span>
              </div>
            </div>
            <label className="flex items-center gap-2 text-sm pb-1">
              <input
                type="checkbox"
                checked={newAsMain}
                onChange={(e) => setNewAsMain(e.target.checked)}
              />
              Set as main model
            </label>
            <button
              onClick={handleCreate}
              disabled={creating || newExpId === ''}
              className="px-4 py-1.5 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50"
            >
              {creating ? 'Adding…' : 'Add to schedule'}
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded p-3 mb-4 text-sm">
          {error}
        </div>
      )}

      {loading && schedules.length === 0 ? (
        <div className="text-gray-500 text-sm">Loading…</div>
      ) : schedules.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center text-gray-500 text-sm">
          No models scheduled yet. Pick one above to start tracking daily predictions.
        </div>
      ) : (
        <div className="space-y-4">
          {schedules.map((s) => {
            const perf = perfById[s.id];
            const lr = s.last_run;
            return (
              <div key={s.id} className="bg-white rounded-lg shadow">
                <div className="p-4 sm:p-5">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2 flex-wrap">
                        <h3 className="font-semibold">
                          {s.experiment_name || `Experiment ${s.experiment_id}`}
                        </h3>
                        {s.is_main && (
                          <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded">
                            MAIN
                          </span>
                        )}
                        {!s.enabled && (
                          <span className="text-xs px-2 py-0.5 bg-gray-200 text-gray-600 rounded">
                            disabled
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Runs daily at{' '}
                        <span className="font-mono">
                          {pad(s.cron_hour)}:{pad(s.cron_minute)}
                        </span>{' '}
                        {s.timezone} · scrapes {s.predict_days_ahead === 0 ? 'today' : `today + ${s.predict_days_ahead}d`}
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <button
                        onClick={() => handleRun(s.id)}
                        disabled={busyRow === s.id}
                        className="px-3 py-1.5 bg-green-600 text-white rounded text-xs hover:bg-green-700 disabled:opacity-50"
                      >
                        {busyRow === s.id ? '…' : 'Run now'}
                      </button>
                      <button
                        onClick={() => patchSchedule(s.id, { enabled: !s.enabled })}
                        disabled={busyRow === s.id}
                        className="px-3 py-1.5 bg-gray-200 rounded text-xs hover:bg-gray-300 disabled:opacity-50"
                      >
                        {s.enabled ? 'Disable' : 'Enable'}
                      </button>
                      {!s.is_main && (
                        <button
                          onClick={() => patchSchedule(s.id, { is_main: true })}
                          disabled={busyRow === s.id}
                          className="px-3 py-1.5 bg-blue-100 text-blue-700 rounded text-xs hover:bg-blue-200 disabled:opacity-50"
                        >
                          Set as main
                        </button>
                      )}
                      <button
                        onClick={() => handleDelete(s.id)}
                        disabled={busyRow === s.id}
                        className="px-3 py-1.5 bg-red-100 text-red-700 rounded text-xs hover:bg-red-200 disabled:opacity-50"
                      >
                        Remove
                      </button>
                    </div>
                  </div>

                  {/* Editable settings */}
                  <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                    <label className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Hour</span>
                      <input
                        type="number"
                        min={0}
                        max={23}
                        className="border rounded px-2 py-1 text-sm"
                        value={s.cron_hour}
                        onChange={(e) =>
                          patchSchedule(s.id, { cron_hour: Number(e.target.value) })
                        }
                      />
                    </label>
                    <label className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Minute</span>
                      <input
                        type="number"
                        min={0}
                        max={59}
                        className="border rounded px-2 py-1 text-sm"
                        value={s.cron_minute}
                        onChange={(e) =>
                          patchSchedule(s.id, { cron_minute: Number(e.target.value) })
                        }
                      />
                    </label>
                    <label className="flex flex-col">
                      <span className="text-xs text-gray-500 mb-1">Days ahead</span>
                      <input
                        type="number"
                        min={0}
                        max={7}
                        className="border rounded px-2 py-1 text-sm"
                        value={s.predict_days_ahead}
                        onChange={(e) =>
                          patchSchedule(s.id, { predict_days_ahead: Number(e.target.value) })
                        }
                      />
                    </label>
                    <label className="flex items-center gap-2 self-end">
                      <input
                        type="checkbox"
                        checked={s.scrape_upcoming}
                        onChange={(e) =>
                          patchSchedule(s.id, { scrape_upcoming: e.target.checked })
                        }
                      />
                      <span className="text-xs">Scrape cards before predicting</span>
                    </label>
                  </div>

                  {/* Last run + performance */}
                  <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">Last run</div>
                      {lr ? (
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span
                              className={`text-xs px-2 py-0.5 rounded ${statusBadge(lr.status)}`}
                            >
                              {lr.status}
                            </span>
                            <span className="text-xs text-gray-500">
                              {lr.run_date} · {lr.trigger}
                            </span>
                          </div>
                          <div className="text-sm mt-1">
                            {lr.races_predicted} predicted, {lr.races_skipped} skipped,{' '}
                            {lr.predictions_written} rows
                          </div>
                          {lr.error_message && (
                            <div className="text-xs text-red-600 mt-1 break-words">
                              {lr.error_message}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="text-sm text-gray-400">Not run yet</div>
                      )}
                    </div>

                    <div className="bg-gray-50 rounded p-3">
                      <div className="text-xs text-gray-500 mb-1">
                        Last 30 days (vs scraped results)
                      </div>
                      {perf ? (
                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div>
                            <div className="text-xs text-gray-500">Races</div>
                            <div className="font-semibold">{perf.races_evaluated}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Top-1 acc</div>
                            <div className="font-semibold">
                              {perf.top1_accuracy != null
                                ? `${(perf.top1_accuracy * 100).toFixed(1)}%`
                                : '—'}
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500">Top-3 hit</div>
                            <div className="font-semibold">
                              {perf.top3_hit_rate != null
                                ? `${(perf.top3_hit_rate * 100).toFixed(1)}%`
                                : '—'}
                            </div>
                          </div>
                          <div className="col-span-3">
                            <div className="text-xs text-gray-500">Mean log-loss</div>
                            <div className="font-semibold">
                              {perf.mean_log_loss != null
                                ? perf.mean_log_loss.toFixed(3)
                                : '—'}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="text-sm text-gray-400">Loading…</div>
                      )}
                    </div>
                  </div>

                  {/* Bankroll placeholder */}
                  <div className="mt-3 text-xs text-gray-400">
                    Per-model virtual bankroll comparison: coming soon.
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
