import { useEffect, useState } from 'react';
import api from '../api/client';
import type {
  Experiment,
  ForecastCombo,
  Track,
  TrioCombo,
} from '../types/models';

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
  race_entry_id?: number;
  dog_name: string;
  trap: number;
  win_probability: number | null;
  place_probability?: number | null;
  show_probability?: number | null;
  predicted_position: number | null;
  predicted_time: number | null;
  confidence: number | null;
  confidence_tier: string | null;
  margin: number | null;
  entropy: number | null;
  edge: number | null;
  is_value: boolean | null;
  kelly: KellyInfo | null;
  sp_decimal_at_pred?: number | null;
  forecast_combos?: ForecastCombo[];
  trio_combos?: TrioCombo[];
}

// Mirrors backend `_compute_kelly_stake` so live edits to the market-odds
// input update the BET/PASS verdict instantly without a server round-trip.
// Keep these defaults identical to the Python side.
const KELLY_FRACTION = 0.25;
const MIN_EDGE = 0.05;
const MAX_STAKE_PCT = 0.05;

function computeKelly(
  winProb: number | null | undefined,
  oddsDecimal: number | null | undefined,
  bankroll: number,
): KellyInfo {
  if (winProb == null) return { bet: false, reason: 'no_probability' };
  if (oddsDecimal == null || oddsDecimal <= 1.0) {
    return { bet: false, reason: 'no_odds' };
  }
  const impliedProb = 1.0 / oddsDecimal;
  const edge = winProb - impliedProb;
  if (edge < MIN_EDGE) {
    return {
      bet: false,
      reason: 'insufficient_edge',
      edge: round4(edge),
      implied_prob: round4(impliedProb),
    };
  }
  const b = oddsDecimal - 1;
  const fStar = (b * winProb - (1 - winProb)) / b;
  const fractionalKelly = Math.max(0, fStar * KELLY_FRACTION);
  const stakePct = Math.min(fractionalKelly, MAX_STAKE_PCT);
  const stake = Math.round(bankroll * stakePct * 100) / 100;
  return {
    bet: true,
    stake,
    stake_pct: Math.round(stakePct * 10000) / 100,
    full_kelly_pct: Math.round(fStar * 10000) / 100,
    edge: round4(edge),
    implied_prob: round4(impliedProb),
    expected_value: round4(winProb * (oddsDecimal - 1) - (1 - winProb)),
  };
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function verdictReason(
  kelly: KellyInfo,
  winProb: number | null | undefined,
  odds: number | null | undefined,
): string {
  if (winProb == null) return 'no model probability';
  if (odds == null || odds <= 1) return 'enter market odds →';
  if (kelly.reason === 'insufficient_edge' && kelly.edge != null) {
    const edgePct = (kelly.edge * 100).toFixed(1);
    return `PASS — only ${edgePct}% edge (need 5%)`;
  }
  return 'PASS';
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
  from_cache?: boolean;
  last_predicted_at?: string | null;
}

interface HistorySession {
  experiment_id: number;
  experiment_name: string | null;
  experiment_algorithm: string | null;
  race_id: number;
  race_date: string | null;
  race_number: number | null;
  race_status: string | null;
  track_name: string | null;
  track_code: string | null;
  distance_m: number | null;
  grade: string | null;
  n_predictions: number;
  last_predicted_at: string | null;
  top_pick: {
    dog_name: string;
    trap: number;
    win_probability: number | null;
    confidence: number | null;
    confidence_tier: string | null;
    edge: number | null;
    is_value: boolean | null;
    kelly_bet: boolean | null;
    kelly_stake: number | null;
    bankroll_used: number | null;
    actual_position: number | null;
    top_pick_won: boolean | null;
  } | null;
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

interface ByDateResponse {
  race_date: string;
  experiment_id: number;
  races_predicted: number;
  races_failed: number;
  races: RacePrediction[];
  errors: { race_id: number; track_code: string; race_number: number; error: string }[];
}

interface ScrapeStatus {
  log_id: number;
  spider_name: string;
  status: string;
  records_scraped: number;
  records_new: number;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
}

interface CardsStatusTrack {
  code: string;
  name: string;
  race_count: number;
  scheduled_count: number;
  resulted_count: number;
  last_scraped_at: string | null;
}

interface CardsStatusLog {
  id: number;
  status: string;
  records_scraped: number;
  records_new: number;
  started_at: string | null;
  completed_at: string | null;
}

interface CardsStatusResponse {
  race_date: string;
  total_races: number;
  tracks: CardsStatusTrack[];
  recent_scrape_logs: CardsStatusLog[];
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
  const [tab, setTab] = useState<'predict' | 'date' | 'history' | 'compare'>('predict');

  // History tab state
  const [history, setHistory] = useState<HistorySession[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyAllModels, setHistoryAllModels] = useState(false);
  const [historyDateFrom, setHistoryDateFrom] = useState('');
  const [historyDateTo, setHistoryDateTo] = useState('');

  // User-entered live odds per race_entry_id, used for the on-the-fly
  // Kelly recompute in the Predict Race results table. Reset whenever a
  // new race is loaded — values from a previous race shouldn't leak.
  const [liveOdds, setLiveOdds] = useState<Record<number, string>>({});
  const [savingOdds, setSavingOdds] = useState(false);

  useEffect(() => {
    if (!predictions) {
      setLiveOdds({});
      return;
    }
    const initial: Record<number, string> = {};
    predictions.predictions.forEach((p) => {
      if (p.race_entry_id == null) return;
      // Seed from kelly.implied_prob (= 1/odds) if the saved Kelly already
      // had a price, falling back to the SP snapshot — so users see the
      // model's last-known odds and can edit from there.
      const fromKelly = p.kelly?.implied_prob
        ? 1 / p.kelly.implied_prob
        : null;
      const seed = fromKelly ?? p.sp_decimal_at_pred ?? null;
      initial[p.race_entry_id] = seed ? seed.toFixed(2) : '';
    });
    setLiveOdds(initial);
  }, [predictions?.race_id, predictions?.last_predicted_at]);

  const handleSaveOdds = async () => {
    if (!predictions || !selectedExp) return;
    const oddsPayload = predictions.predictions
      .filter((p) => p.race_entry_id != null)
      .map((p) => {
        const raw = liveOdds[p.race_entry_id!];
        const parsed = raw ? parseFloat(raw) : null;
        return {
          race_entry_id: p.race_entry_id!,
          decimal_odds: parsed && parsed > 1 ? parsed : null,
        };
      });
    setSavingOdds(true);
    try {
      const res = await api.post<RacePrediction>(
        `/predictions/race/${predictions.race_id}/odds`,
        {
          experiment_id: selectedExp,
          bankroll,
          odds: oddsPayload,
        },
      );
      setPredictions(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to save odds');
    } finally {
      setSavingOdds(false);
    }
  };

  // Race picker state
  const [raceDate, setRaceDate] = useState(new Date().toISOString().split('T')[0]);
  const [trackCode, setTrackCode] = useState('');
  const [availableRaces, setAvailableRaces] = useState<RaceOption[]>([]);
  const [selectedRaceId, setSelectedRaceId] = useState<number | null>(null);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const [bankroll, setBankroll] = useState(100);

  // Manual race ID fallback
  const [manualRaceId, setManualRaceId] = useState('');

  // Date-based scrape + predict flow state
  const [scrapeDate, setScrapeDate] = useState(new Date().toISOString().split('T')[0]);
  const [includeFormDetail, setIncludeFormDetail] = useState(false);
  const [scrapeStatus, setScrapeStatus] = useState<ScrapeStatus | null>(null);
  const [scrapeRunning, setScrapeRunning] = useState(false);
  const [byDateResult, setByDateResult] = useState<ByDateResponse | null>(null);
  const [byDateLoading, setByDateLoading] = useState(false);
  const [expandedRaceId, setExpandedRaceId] = useState<number | null>(null);
  const [cardsStatus, setCardsStatus] = useState<CardsStatusResponse | null>(null);

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

  const handlePredict = async (forceRefresh: boolean = false) => {
    const raceId = selectedRaceId || (manualRaceId ? parseInt(manualRaceId) : null);
    if (!selectedExp || !raceId) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({
        experiment_id: String(selectedExp),
        bankroll: String(bankroll),
      });
      if (forceRefresh) params.set('refresh', 'true');
      const res = await api.get(`/predictions/race/${raceId}?${params}`);
      setPredictions(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to generate predictions');
    }
    setLoading(false);
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const params = new URLSearchParams({ limit: '200' });
      if (!historyAllModels && selectedExp) {
        params.set('experiment_id', String(selectedExp));
      }
      if (historyDateFrom) params.set('race_date_from', historyDateFrom);
      if (historyDateTo) params.set('race_date_to', historyDateTo);
      const res = await api.get<{ sessions: HistorySession[]; total: number }>(
        `/predictions/history?${params}`,
      );
      setHistory(res.data.sessions);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to load history');
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const openHistoryEntry = async (s: HistorySession) => {
    if (!s.race_id || !s.experiment_id) return;
    setSelectedExp(s.experiment_id);
    setSelectedRaceId(s.race_id);
    setManualRaceId('');
    setLoading(true);
    setTab('predict');
    try {
      const res = await api.get<RacePrediction>(
        `/predictions/race/${s.race_id}/saved?experiment_id=${s.experiment_id}`,
      );
      setPredictions(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to load saved prediction');
    } finally {
      setLoading(false);
    }
  };

  // Fetch which tracks already have race cards scraped for the selected date
  const refreshCardsStatus = async (d: string) => {
    if (!d) return;
    try {
      const res = await api.get<CardsStatusResponse>(
        `/scraping/cards-status?race_date=${d}`,
      );
      setCardsStatus(res.data);
    } catch {
      setCardsStatus(null);
    }
  };

  useEffect(() => {
    if (tab !== 'date' || !scrapeDate) return;
    refreshCardsStatus(scrapeDate);
  }, [tab, scrapeDate]);

  // Poll the scrape log while a card scrape is running
  useEffect(() => {
    if (!scrapeStatus || !scrapeRunning) return;
    interface RawLog {
      id: number;
      spider_name: string;
      status: string;
      records_scraped: number;
      records_new: number;
      error_message: string | null;
      started_at: string | null;
      completed_at: string | null;
    }
    const interval = setInterval(async () => {
      try {
        const res = await api.get<{ recent_logs: RawLog[] }>('/scraping/status');
        const latest = res.data.recent_logs.find(l => l.id === scrapeStatus.log_id);
        if (latest) {
          setScrapeStatus({
            log_id: latest.id,
            spider_name: latest.spider_name,
            status: latest.status,
            records_scraped: latest.records_scraped,
            records_new: latest.records_new,
            error_message: latest.error_message,
            started_at: latest.started_at,
            completed_at: latest.completed_at,
          });
          if (latest.status !== 'running') {
            setScrapeRunning(false);
            refreshCardsStatus(scrapeDate);
          }
        }
      } catch {
        // ignore transient polling failures
      }
    }, 2500);
    return () => clearInterval(interval);
  }, [scrapeStatus?.log_id, scrapeRunning, scrapeDate]);

  const handleStartScrape = async () => {
    setScrapeStatus(null);
    setByDateResult(null);
    try {
      const res = await api.post<{ message: string; log_id: number | null }>(
        '/scraping/scrape-date',
        { race_date: scrapeDate, include_form_detail: includeFormDetail },
      );
      if (res.data.log_id == null) {
        alert(res.data.message);
        return;
      }
      setScrapeStatus({
        log_id: res.data.log_id,
        spider_name: 'gri-card',
        status: 'running',
        records_scraped: 0,
        records_new: 0,
        error_message: null,
        started_at: new Date().toISOString(),
        completed_at: null,
      });
      setScrapeRunning(true);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to start scrape');
    }
  };

  const handlePredictDate = async () => {
    if (!selectedExp) return;
    setByDateLoading(true);
    setByDateResult(null);
    try {
      const params = new URLSearchParams({
        race_date: scrapeDate,
        experiment_id: String(selectedExp),
        bankroll: String(bankroll),
        only_scheduled: 'true',
      });
      const res = await api.get<ByDateResponse>(`/predictions/by-date?${params}`);
      setByDateResult(res.data);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to predict races for date');
    }
    setByDateLoading(false);
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
      <h1 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6">Predictions</h1>

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
              onClick={() => setTab('date')}
              className={`px-5 py-3 text-sm font-medium border-b-2 ${
                tab === 'date' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500'
              }`}
            >
              Predict by Date
            </button>
            <button
              onClick={() => { setTab('history'); loadHistory(); }}
              className={`px-5 py-3 text-sm font-medium border-b-2 ${
                tab === 'history' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500'
              }`}
            >
              History
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
                      onClick={() => handlePredict(false)}
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
                        <div className="flex items-center gap-2 mt-1 flex-wrap">
                          {predictions.from_cache ? (
                            <span className="inline-flex items-center gap-1 text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
                              Saved prediction
                              {predictions.last_predicted_at && (
                                <span className="text-blue-500">
                                  · {new Date(predictions.last_predicted_at).toLocaleString()}
                                </span>
                              )}
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-xs bg-green-50 text-green-700 px-2 py-0.5 rounded-full">
                              Fresh prediction
                            </span>
                          )}
                          <button
                            onClick={() => handlePredict(true)}
                            disabled={loading}
                            className="text-xs text-blue-600 hover:text-blue-800 underline disabled:opacity-50"
                            title="Recompute from the model (e.g. new bankroll or after retraining)"
                          >
                            {loading ? 'Refreshing...' : 'Refresh prediction'}
                          </button>
                        </div>
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
                  <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
                    <div className="px-4 pt-3 pb-2 border-b text-xs text-gray-500 flex items-center justify-between gap-2 flex-wrap">
                      <span>
                        Type the current market odds for each dog and the
                        verdict updates instantly. Save to persist them.
                      </span>
                      <button
                        onClick={handleSaveOdds}
                        disabled={savingOdds}
                        className="bg-blue-600 text-white px-3 py-1.5 rounded-md text-xs font-medium hover:bg-blue-700 disabled:opacity-50"
                      >
                        {savingOdds ? 'Saving...' : 'Save odds & bet plan'}
                      </button>
                    </div>
                    <table className="w-full text-sm text-left min-w-[860px]">
                      <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                        <tr>
                          <th className="px-3 sm:px-4 py-3">Rank</th>
                          <th className="px-3 sm:px-4 py-3">Trap</th>
                          <th className="px-3 sm:px-4 py-3">Dog</th>
                          <th className="px-3 sm:px-4 py-3">Win</th>
                          <th className="px-3 sm:px-4 py-3" title="P(finish 1st or 2nd)">Place</th>
                          <th className="px-3 sm:px-4 py-3" title="P(finish 1st, 2nd, or 3rd)">Show</th>
                          <th className="px-3 sm:px-4 py-3">Market Odds</th>
                          <th className="px-3 sm:px-4 py-3">Edge</th>
                          <th className="px-3 sm:px-4 py-3">Verdict</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y">
                        {predictions.predictions.map((p, i) => {
                          const isTopPick = i === 0;
                          const oddsStr = p.race_entry_id != null
                            ? liveOdds[p.race_entry_id] ?? ''
                            : '';
                          const oddsNum = oddsStr ? parseFloat(oddsStr) : null;
                          const liveKelly = computeKelly(
                            p.win_probability,
                            oddsNum && oddsNum > 1 ? oddsNum : null,
                            bankroll,
                          );
                          const verdictBet = liveKelly.bet;
                          const liveEdge = liveKelly.edge ?? null;
                          return (
                            <tr
                              key={i}
                              className={
                                isTopPick && verdictBet
                                  ? 'bg-green-50'
                                  : isTopPick
                                  ? 'bg-blue-50'
                                  : verdictBet
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
                              <td className="px-4 py-3 font-mono text-gray-700">
                                {p.place_probability != null
                                  ? `${(p.place_probability * 100).toFixed(1)}%`
                                  : '-'}
                              </td>
                              <td className="px-4 py-3 font-mono text-gray-700">
                                {p.show_probability != null
                                  ? `${(p.show_probability * 100).toFixed(1)}%`
                                  : '-'}
                              </td>
                              <td className="px-4 py-3">
                                <input
                                  type="number"
                                  step="0.01"
                                  min="1.01"
                                  inputMode="decimal"
                                  value={oddsStr}
                                  onChange={(e) => {
                                    if (p.race_entry_id == null) return;
                                    setLiveOdds((prev) => ({
                                      ...prev,
                                      [p.race_entry_id!]: e.target.value,
                                    }));
                                  }}
                                  placeholder="e.g. 4.50"
                                  className="w-20 border rounded-md px-2 py-1 text-sm font-mono"
                                />
                              </td>
                              <td className="px-4 py-3 font-mono text-sm">
                                {edgeDisplay(liveEdge)}
                              </td>
                              <td className="px-4 py-3 text-sm">
                                {verdictBet ? (
                                  <div>
                                    <span className="inline-block bg-green-600 text-white px-2 py-0.5 rounded font-bold text-xs">
                                      BET ${liveKelly.stake?.toFixed(2)}
                                    </span>
                                    <span className="text-gray-400 text-xs ml-1">
                                      ({liveKelly.stake_pct}% of bank)
                                    </span>
                                  </div>
                                ) : (
                                  <span className="text-gray-500 text-xs">
                                    {verdictReason(liveKelly, p.win_probability, oddsNum)}
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Live betting recommendation — recomputes against
                      whatever odds the user has typed in. */}
                  {(() => {
                    const liveBets = predictions.predictions
                      .map((p) => {
                        const oddsStr = p.race_entry_id != null
                          ? liveOdds[p.race_entry_id]
                          : '';
                        const odds = oddsStr ? parseFloat(oddsStr) : null;
                        const kelly = computeKelly(
                          p.win_probability,
                          odds && odds > 1 ? odds : null,
                          bankroll,
                        );
                        return { p, kelly, odds };
                      })
                      .filter((x) => x.kelly.bet);

                    if (liveBets.length > 0) {
                      const totalStake = liveBets.reduce(
                        (sum, x) => sum + (x.kelly.stake || 0), 0,
                      );
                      const totalEv = liveBets.reduce(
                        (sum, x) => sum + ((x.kelly.expected_value || 0) * (x.kelly.stake || 0)), 0,
                      );
                      return (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                          <h3 className="font-semibold text-green-800 mb-2">
                            Recommended Bets ({liveBets.length})
                          </h3>
                          <div className="space-y-2">
                            {liveBets.map(({ p, kelly, odds }, idx) => (
                              <div key={idx} className="flex items-center justify-between text-sm border-b border-green-100 last:border-0 pb-2 last:pb-0">
                                <div>
                                  <span className="font-semibold text-green-900">
                                    {p.dog_name}
                                  </span>
                                  <span className="text-green-700 text-xs ml-2">
                                    Trap {p.trap} · win {((p.win_probability || 0) * 100).toFixed(1)}% @ {odds?.toFixed(2)}
                                  </span>
                                </div>
                                <div className="text-right">
                                  <p className="font-mono font-semibold text-green-900">
                                    ${kelly.stake?.toFixed(2)}
                                  </p>
                                  <p className="text-green-600 text-xs">
                                    edge {((kelly.edge || 0) * 100).toFixed(1)}% · EV ${((kelly.expected_value || 0) * (kelly.stake || 0)).toFixed(2)}
                                  </p>
                                </div>
                              </div>
                            ))}
                          </div>
                          <div className="mt-3 pt-2 border-t border-green-200 flex justify-between text-xs text-green-700">
                            <span>
                              Total stake: <span className="font-mono font-semibold">${totalStake.toFixed(2)}</span>
                              {' '}({((totalStake / bankroll) * 100).toFixed(1)}% of ${bankroll} bankroll)
                            </span>
                            <span>
                              Total EV: <span className="font-mono font-semibold">${totalEv.toFixed(2)}</span>
                            </span>
                          </div>
                        </div>
                      );
                    }

                    const anyOdds = predictions.predictions.some((p) => {
                      const o = p.race_entry_id != null ? liveOdds[p.race_entry_id] : '';
                      return o && parseFloat(o) > 1;
                    });
                    return (
                      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <p className="text-gray-600 text-sm">
                          {anyOdds
                            ? 'No recommended bets at the current odds — none clear the 5% edge threshold.'
                            : 'Enter the market odds for each dog above to get a bet recommendation.'}
                          {predictions.predictions[0]?.confidence_tier === 'avoid' &&
                            ' The model also has low confidence in this race overall.'}
                        </p>
                      </div>
                    );
                  })()}

                  {/* Forecast (1st+2nd) and Trio (1st+2nd+3rd) combos —
                      Henery-discounted Plackett-Luce expansion of the
                      win-prob layer. Probabilities here are model
                      estimates only; pair them against Tote / Betfair
                      F/C dividends in your bookmaker before staking. */}
                  {(() => {
                    const top = predictions.predictions[0];
                    const fc = top?.forecast_combos || [];
                    const trio = top?.trio_combos || [];
                    if (fc.length === 0 && trio.length === 0) return null;

                    const fmtPct = (p: number) => `${(p * 100).toFixed(2)}%`;
                    const fmtFairOdds = (p: number) =>
                      p > 0 ? (1 / p).toFixed(2) : '-';

                    return (
                      <div className="bg-white rounded-lg shadow p-4 space-y-4">
                        <div>
                          <h3 className="font-semibold text-gray-800 mb-1">
                            Forecast & Trio combos
                          </h3>
                          <p className="text-xs text-gray-500">
                            Top 1st+2nd and 1st+2nd+3rd combinations from
                            the ordering model. "Fair odds" is 1 /
                            probability — your bookmaker's dividend
                            needs to clear that (plus margin) for the
                            bet to have positive EV.
                          </p>
                        </div>

                        {fc.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-gray-700 mb-1">
                              Top forecasts (1st &amp; 2nd)
                            </h4>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs text-left">
                                <thead className="bg-gray-50 text-gray-500 uppercase">
                                  <tr>
                                    <th className="px-2 py-1.5">#</th>
                                    <th className="px-2 py-1.5">1st</th>
                                    <th className="px-2 py-1.5">2nd</th>
                                    <th className="px-2 py-1.5 text-right">P</th>
                                    <th className="px-2 py-1.5 text-right">Fair odds</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y">
                                  {fc.slice(0, 10).map((c, i) => (
                                    <tr key={i} className="hover:bg-gray-50">
                                      <td className="px-2 py-1.5 text-gray-400">{i + 1}</td>
                                      <td className="px-2 py-1.5">
                                        <span className="font-medium">{c.first_dog || `entry ${c.first_entry_id}`}</span>
                                        {c.first_trap != null && (
                                          <span className="text-gray-400 ml-1">(T{c.first_trap})</span>
                                        )}
                                      </td>
                                      <td className="px-2 py-1.5">
                                        <span className="font-medium">{c.second_dog || `entry ${c.second_entry_id}`}</span>
                                        {c.second_trap != null && (
                                          <span className="text-gray-400 ml-1">(T{c.second_trap})</span>
                                        )}
                                      </td>
                                      <td className="px-2 py-1.5 text-right font-mono">{fmtPct(c.probability)}</td>
                                      <td className="px-2 py-1.5 text-right font-mono text-gray-600">{fmtFairOdds(c.probability)}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}

                        {trio.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-gray-700 mb-1">
                              Top trios (1st, 2nd &amp; 3rd)
                            </h4>
                            <div className="overflow-x-auto">
                              <table className="w-full text-xs text-left">
                                <thead className="bg-gray-50 text-gray-500 uppercase">
                                  <tr>
                                    <th className="px-2 py-1.5">#</th>
                                    <th className="px-2 py-1.5">1st</th>
                                    <th className="px-2 py-1.5">2nd</th>
                                    <th className="px-2 py-1.5">3rd</th>
                                    <th className="px-2 py-1.5 text-right">P</th>
                                    <th className="px-2 py-1.5 text-right">Fair odds</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y">
                                  {trio.slice(0, 10).map((c, i) => (
                                    <tr key={i} className="hover:bg-gray-50">
                                      <td className="px-2 py-1.5 text-gray-400">{i + 1}</td>
                                      <td className="px-2 py-1.5">
                                        <span className="font-medium">{c.first_dog || `entry ${c.first_entry_id}`}</span>
                                        {c.first_trap != null && (
                                          <span className="text-gray-400 ml-1">(T{c.first_trap})</span>
                                        )}
                                      </td>
                                      <td className="px-2 py-1.5">
                                        <span className="font-medium">{c.second_dog || `entry ${c.second_entry_id}`}</span>
                                        {c.second_trap != null && (
                                          <span className="text-gray-400 ml-1">(T{c.second_trap})</span>
                                        )}
                                      </td>
                                      <td className="px-2 py-1.5">
                                        <span className="font-medium">{c.third_dog || `entry ${c.third_entry_id}`}</span>
                                        {c.third_trap != null && (
                                          <span className="text-gray-400 ml-1">(T{c.third_trap})</span>
                                        )}
                                      </td>
                                      <td className="px-2 py-1.5 text-right font-mono">{fmtPct(c.probability)}</td>
                                      <td className="px-2 py-1.5 text-right font-mono text-gray-600">{fmtFairOdds(c.probability)}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          )}

          {tab === 'date' && (
            <div>
              {/* Step 1: scrape upcoming cards for the date */}
              <div className="bg-white rounded-lg shadow p-5 mb-4">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="font-semibold">Step 1 — Scrape upcoming cards</h2>
                  <span className="text-xs text-gray-400">brute-forces all active GRI tracks</span>
                </div>
                <div className="flex gap-3 items-end flex-wrap">
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">Race date</label>
                    <input
                      type="date"
                      value={scrapeDate}
                      onChange={(e) => setScrapeDate(e.target.value)}
                      className="border rounded-md px-3 py-2 text-sm"
                    />
                  </div>
                  <label className="flex items-center gap-2 text-sm text-gray-600 pb-2">
                    <input
                      type="checkbox"
                      checked={includeFormDetail}
                      onChange={(e) => setIncludeFormDetail(e.target.checked)}
                    />
                    Include form detail (trainer / sire / dam)
                  </label>
                  <button
                    onClick={handleStartScrape}
                    disabled={scrapeRunning || !scrapeDate}
                    className="bg-blue-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
                  >
                    {scrapeRunning ? 'Scraping...' : 'Scrape this date'}
                  </button>
                </div>
                {scrapeStatus && (
                  <div className="mt-3 text-xs">
                    <span className={`inline-block px-2 py-0.5 rounded-full mr-2 ${
                      scrapeStatus.status === 'success' ? 'bg-green-100 text-green-700'
                      : scrapeStatus.status === 'failed' ? 'bg-red-100 text-red-700'
                      : scrapeStatus.status === 'partial' ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-blue-100 text-blue-700'
                    }`}>
                      {scrapeStatus.status.toUpperCase()}
                    </span>
                    <span className="text-gray-600">
                      {scrapeStatus.records_scraped} race(s) scraped, {scrapeStatus.records_new} new
                    </span>
                    {scrapeStatus.error_message && (
                      <p className="text-gray-500 mt-1">{scrapeStatus.error_message}</p>
                    )}
                  </div>
                )}

                {/* Already-scraped tracks for the selected date */}
                <div className="mt-4 pt-3 border-t">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
                      Already scraped for {scrapeDate || '—'}
                    </h3>
                    {cardsStatus && (
                      <span className="text-xs text-gray-400">
                        {cardsStatus.tracks.length} track(s) · {cardsStatus.total_races} race(s)
                      </span>
                    )}
                  </div>
                  {!cardsStatus || cardsStatus.tracks.length === 0 ? (
                    <p className="text-xs text-gray-400">
                      No cards scraped for this date yet — click "Scrape this date" above.
                    </p>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {cardsStatus.tracks.map(t => {
                        const fullyResulted = t.resulted_count > 0 && t.scheduled_count === 0;
                        const partiallyResulted = t.resulted_count > 0 && t.scheduled_count > 0;
                        const tooltip = [
                          `${t.race_count} race(s)`,
                          t.scheduled_count > 0 ? `${t.scheduled_count} scheduled` : null,
                          t.resulted_count > 0 ? `${t.resulted_count} resulted` : null,
                          t.last_scraped_at
                            ? `last scraped ${new Date(t.last_scraped_at).toLocaleString()}`
                            : null,
                        ].filter(Boolean).join(' · ');
                        return (
                          <span
                            key={t.code}
                            title={tooltip}
                            className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs border ${
                              fullyResulted
                                ? 'bg-gray-50 border-gray-200 text-gray-600'
                                : partiallyResulted
                                ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
                                : 'bg-green-50 border-green-200 text-green-800'
                            }`}
                          >
                            <span className="font-medium">{t.name}</span>
                            <span className="font-mono text-[11px] opacity-70">
                              {t.race_count}
                            </span>
                          </span>
                        );
                      })}
                    </div>
                  )}
                  {cardsStatus && cardsStatus.recent_scrape_logs.length > 0 && (
                    <details className="mt-2">
                      <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                        Recent scrape jobs for this date ({cardsStatus.recent_scrape_logs.length})
                      </summary>
                      <ul className="mt-1 space-y-1 text-xs text-gray-500">
                        {cardsStatus.recent_scrape_logs.map(l => (
                          <li key={l.id} className="font-mono">
                            #{l.id} {l.status} — {l.records_scraped} scraped, {l.records_new} new
                            {l.completed_at && ` · ${new Date(l.completed_at).toLocaleString()}`}
                          </li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              </div>

              {/* Step 2: predict all scheduled races on the date */}
              <div className="bg-white rounded-lg shadow p-5 mb-4">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="font-semibold">Step 2 — Predict scheduled races</h2>
                  <span className="text-xs text-gray-400">uses the model selected above</span>
                </div>
                <button
                  onClick={handlePredictDate}
                  disabled={byDateLoading || !selectedExp || !scrapeDate}
                  className="bg-green-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-green-700 disabled:opacity-50"
                >
                  {byDateLoading ? 'Predicting...' : 'Predict all races on this date'}
                </button>
              </div>

              {/* Results */}
              {byDateResult && (
                <div className="space-y-3">
                  <div className="bg-white rounded-lg shadow p-4 text-sm flex items-center gap-4 flex-wrap">
                    <span className="text-gray-700 font-medium">{byDateResult.race_date}</span>
                    <span className="text-green-700">{byDateResult.races_predicted} predicted</span>
                    {byDateResult.races_failed > 0 && (
                      <span className="text-red-600">{byDateResult.races_failed} failed</span>
                    )}
                  </div>

                  {byDateResult.races.length === 0 ? (
                    <div className="bg-white rounded-lg shadow p-6 text-center text-gray-500 text-sm">
                      No scheduled races found for this date. Did you run Step 1 first?
                    </div>
                  ) : (
                    Object.entries(
                      byDateResult.races.reduce((acc, r) => {
                        (acc[r.track_name] = acc[r.track_name] || []).push(r);
                        return acc;
                      }, {} as Record<string, RacePrediction[]>)
                    ).map(([trackName, trackRaces]) => (
                      <div key={trackName} className="bg-white rounded-lg shadow overflow-hidden">
                        <div className="px-5 py-3 bg-gray-50 border-b">
                          <h3 className="font-semibold text-sm">{trackName}</h3>
                          <p className="text-xs text-gray-500">{trackRaces.length} race(s)</p>
                        </div>
                        <div className="divide-y">
                          {trackRaces.map(r => {
                            const top = r.predictions[0];
                            const expanded = expandedRaceId === r.race_id;
                            return (
                              <div key={r.race_id}>
                                <button
                                  onClick={() => setExpandedRaceId(expanded ? null : r.race_id)}
                                  className="w-full text-left px-5 py-3 hover:bg-gray-50 flex items-center gap-4 text-sm"
                                >
                                  <span className="font-mono w-10">R{r.race_number}</span>
                                  <span className="text-gray-500 w-20">{r.distance_m}m {r.grade}</span>
                                  {top && (
                                    <>
                                      <span className="font-medium flex-1">{top.dog_name} (T{top.trap})</span>
                                      <span className={`font-mono ${probColor(top.win_probability)}`}>
                                        {top.win_probability ? `${(top.win_probability * 100).toFixed(1)}%` : '-'}
                                      </span>
                                      {tierBadge(top.confidence_tier)}
                                      {top.kelly?.bet && (
                                        <span className="text-green-700 font-medium">
                                          ${top.kelly.stake?.toFixed(2)}
                                        </span>
                                      )}
                                    </>
                                  )}
                                  <span className="text-gray-400">{expanded ? '▾' : '▸'}</span>
                                </button>
                                {expanded && (
                                  <div className="px-5 pb-4 bg-gray-50">
                                    <table className="w-full text-xs">
                                      <thead className="text-gray-500 uppercase">
                                        <tr>
                                          <th className="text-left py-1">Rank</th>
                                          <th className="text-left py-1">Trap</th>
                                          <th className="text-left py-1">Dog</th>
                                          <th className="text-left py-1">Win Prob</th>
                                          <th className="text-left py-1">Edge</th>
                                          <th className="text-left py-1">Stake</th>
                                        </tr>
                                      </thead>
                                      <tbody className="divide-y">
                                        {r.predictions.map((p, i) => (
                                          <tr key={i} className={i === 0 ? 'bg-blue-50/50' : ''}>
                                            <td className="py-1 pr-2">{i + 1}</td>
                                            <td className="py-1 pr-2">{p.trap}</td>
                                            <td className="py-1 pr-2 font-medium">{p.dog_name}</td>
                                            <td className={`py-1 pr-2 font-mono ${probColor(p.win_probability)}`}>
                                              {p.win_probability ? `${(p.win_probability * 100).toFixed(1)}%` : '-'}
                                            </td>
                                            <td className="py-1 pr-2 font-mono">{edgeDisplay(p.edge)}</td>
                                            <td className="py-1 pr-2 font-mono">
                                              {p.kelly?.bet ? `$${p.kelly.stake?.toFixed(2)}` : '-'}
                                            </td>
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))
                  )}

                  {byDateResult.errors.length > 0 && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <h4 className="font-semibold text-red-800 text-sm mb-2">Errors</h4>
                      <ul className="text-xs text-red-700 space-y-1">
                        {byDateResult.errors.map((e, i) => (
                          <li key={i}>
                            {e.track_code} R{e.race_number} (race_id={e.race_id}): {e.error}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {tab === 'history' && (
            <div>
              <div className="bg-white rounded-lg shadow p-5 mb-4">
                <div className="flex items-end gap-3 flex-wrap">
                  <label className="flex items-center gap-2 text-sm text-gray-600 pb-2">
                    <input
                      type="checkbox"
                      checked={historyAllModels}
                      onChange={(e) => setHistoryAllModels(e.target.checked)}
                    />
                    Show all models
                  </label>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">From date</label>
                    <input
                      type="date"
                      value={historyDateFrom}
                      onChange={(e) => setHistoryDateFrom(e.target.value)}
                      className="border rounded-md px-3 py-2 text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">To date</label>
                    <input
                      type="date"
                      value={historyDateTo}
                      onChange={(e) => setHistoryDateTo(e.target.value)}
                      className="border rounded-md px-3 py-2 text-sm"
                    />
                  </div>
                  <button
                    onClick={loadHistory}
                    disabled={historyLoading}
                    className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
                  >
                    {historyLoading ? 'Loading...' : 'Apply filters'}
                  </button>
                </div>
                <p className="text-xs text-gray-400 mt-3">
                  Past predictions are saved automatically. Click any row to reopen the saved prediction without re-running the model.
                </p>
              </div>

              {historyLoading ? (
                <p className="text-gray-500">Loading history...</p>
              ) : history.length === 0 ? (
                <div className="bg-white rounded-lg shadow p-8 text-center">
                  <p className="text-gray-500">No prediction history yet</p>
                  <p className="text-gray-400 text-sm mt-1">
                    Run a prediction in the "Predict Race" or "Predict by Date" tab — they'll show up here automatically.
                  </p>
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
                  <table className="w-full text-sm text-left min-w-[820px]">
                    <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                      <tr>
                        <th className="px-3 sm:px-4 py-3">Predicted at</th>
                        <th className="px-3 sm:px-4 py-3">Race</th>
                        <th className="px-3 sm:px-4 py-3">Track</th>
                        <th className="px-3 sm:px-4 py-3">Model</th>
                        <th className="px-3 sm:px-4 py-3">Top Pick</th>
                        <th className="px-3 sm:px-4 py-3">Win Prob</th>
                        <th className="px-3 sm:px-4 py-3">Confidence</th>
                        <th className="px-3 sm:px-4 py-3">Edge</th>
                        <th className="px-3 sm:px-4 py-3">Bet?</th>
                        <th className="px-3 sm:px-4 py-3">Result</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {history.map((s) => {
                        const tp = s.top_pick;
                        return (
                          <tr
                            key={`${s.experiment_id}-${s.race_id}`}
                            onClick={() => openHistoryEntry(s)}
                            className="cursor-pointer hover:bg-blue-50"
                          >
                            <td className="px-4 py-3 text-xs text-gray-600">
                              {s.last_predicted_at
                                ? new Date(s.last_predicted_at).toLocaleString()
                                : '-'}
                            </td>
                            <td className="px-4 py-3 text-xs">
                              <div className="font-medium">
                                R{s.race_number ?? '-'} {s.distance_m ? `· ${s.distance_m}m` : ''}
                              </div>
                              <div className="text-gray-400">{s.race_date}</div>
                            </td>
                            <td className="px-4 py-3">{s.track_name || s.track_code || '-'}</td>
                            <td className="px-4 py-3 text-xs">
                              <div className="font-medium">{s.experiment_name || `#${s.experiment_id}`}</div>
                              <div className="text-gray-400">{s.experiment_algorithm}</div>
                            </td>
                            <td className="px-4 py-3 font-medium">
                              {tp ? `${tp.dog_name} (T${tp.trap})` : '-'}
                            </td>
                            <td className={`px-4 py-3 font-mono ${probColor(tp?.win_probability ?? null)}`}>
                              {tp?.win_probability != null
                                ? `${(tp.win_probability * 100).toFixed(1)}%`
                                : '-'}
                            </td>
                            <td className="px-4 py-3">{tierBadge(tp?.confidence_tier ?? null) || '-'}</td>
                            <td className="px-4 py-3 font-mono text-xs">
                              {edgeDisplay(tp?.edge ?? null)}
                            </td>
                            <td className="px-4 py-3 text-xs">
                              {tp?.kelly_bet ? (
                                <span className="text-green-700 font-bold">
                                  ${tp.kelly_stake?.toFixed(2)}
                                </span>
                              ) : (
                                <span className="text-gray-300">-</span>
                              )}
                            </td>
                            <td className="px-4 py-3 text-xs">
                              {tp?.top_pick_won === true ? (
                                <span className="text-green-700 font-bold">WON</span>
                              ) : tp?.actual_position != null ? (
                                <span className="text-gray-500">{tp.actual_position}{tp.actual_position === 2 ? 'nd' : tp.actual_position === 3 ? 'rd' : 'th'}</span>
                              ) : s.race_status === 'scheduled' ? (
                                <span className="text-gray-400">scheduled</span>
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
                <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
                  <table className="w-full text-sm text-left min-w-[720px]">
                    <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
                      <tr>
                        <th className="px-3 sm:px-4 py-3">Date</th>
                        <th className="px-3 sm:px-4 py-3">Track</th>
                        <th className="px-3 sm:px-4 py-3">Race</th>
                        <th className="px-3 sm:px-4 py-3">Dog</th>
                        <th className="px-3 sm:px-4 py-3">Trap</th>
                        <th className="px-3 sm:px-4 py-3">Win Prob</th>
                        <th className="px-3 sm:px-4 py-3">Edge</th>
                        <th className="px-3 sm:px-4 py-3">Actual Pos</th>
                        <th className="px-3 sm:px-4 py-3">SP</th>
                        <th className="px-3 sm:px-4 py-3">Value?</th>
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
