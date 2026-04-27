import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import type { Track } from '../types/models';

interface ManualEntry {
  trap: number;
  dog_name: string;
  trainer_name: string;
  weight_kg: string;
  sp_decimal: string;
}

interface ScrapedEntry {
  trap: number;
  dog_name: string;
  trainer_name: string | null;
  sire_name: string | null;
  dam_name: string | null;
  weight_kg: number | null;
}

interface ScrapedRace {
  race_number: number | null;
  race_time: string | null;
  distance_m: number | null;
  grade: string | null;
  race_type: string;
  entries: ScrapedEntry[];
}

interface ScrapeUpcomingResponse {
  track_code: string;
  race_date: string;
  url_used: string | null;
  races_found: number;
  races: ScrapedRace[];
  saved: boolean;
  db_stats: Record<string, number> | null;
  message: string | null;
}

interface ManualRaceOut {
  race_id: number;
  track_id: number;
  track_name: string;
  race_number: number;
  race_date: string;
  entries_created: number;
  dogs_created: number;
  message: string;
}

const blankEntry = (trap: number): ManualEntry => ({
  trap,
  dog_name: '',
  trainer_name: '',
  weight_kg: '',
  sp_decimal: '',
});

const tomorrowISO = () => {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().split('T')[0];
};

export default function RaceCardEntry() {
  const navigate = useNavigate();

  const [tracks, setTracks] = useState<Track[]>([]);
  const [trackCode, setTrackCode] = useState('');
  const [raceDate, setRaceDate] = useState(tomorrowISO());

  // Race-level form fields
  const [raceNumber, setRaceNumber] = useState('1');
  const [raceTime, setRaceTime] = useState('');
  const [distanceM, setDistanceM] = useState('525');
  const [grade, setGrade] = useState('');
  const [going, setGoing] = useState('');

  // Entries (default to 6 traps; user can resize 4-8)
  const [numEntries, setNumEntries] = useState(6);
  const [entries, setEntries] = useState<ManualEntry[]>(() =>
    Array.from({ length: 6 }, (_, i) => blankEntry(i + 1)),
  );

  // Scrape preview
  const [scraping, setScraping] = useState(false);
  const [scrapePreview, setScrapePreview] = useState<ScrapeUpcomingResponse | null>(null);
  const [scrapeMessage, setScrapeMessage] = useState<string>('');

  // Save state
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<ManualRaceOut | null>(null);
  const [saveError, setSaveError] = useState<string>('');

  useEffect(() => {
    api.get<Track[]>('/tracks/').then((res) => {
      setTracks(res.data);
      const firstActive = res.data.find((t) => t.active);
      if (firstActive) setTrackCode(firstActive.code);
    });
  }, []);

  // Adjust the entries array when numEntries changes
  useEffect(() => {
    setEntries((prev) => {
      const next = Array.from({ length: numEntries }, (_, i) =>
        prev[i] ?? blankEntry(i + 1),
      );
      // Renumber traps 1..N to keep them consistent
      return next.map((e, i) => ({ ...e, trap: i + 1 }));
    });
  }, [numEntries]);

  const updateEntry = (idx: number, field: keyof ManualEntry, value: string) => {
    setEntries((prev) => {
      const next = [...prev];
      const current = { ...next[idx] };
      if (field === 'trap') {
        current.trap = parseInt(value) || idx + 1;
      } else {
        current[field] = value;
      }
      next[idx] = current;
      return next;
    });
  };

  const handleScrapePreview = async () => {
    if (!trackCode || !raceDate) return;
    setScraping(true);
    setScrapeMessage('');
    setScrapePreview(null);
    try {
      const res = await api.post<ScrapeUpcomingResponse>('/scraping/upcoming', {
        track_code: trackCode,
        race_date: raceDate,
        save: false,
      });
      setScrapePreview(res.data);
      setScrapeMessage(res.data.message ?? '');
    } catch (err: any) {
      setScrapeMessage(err.response?.data?.detail ?? 'Scrape failed.');
    }
    setScraping(false);
  };

  const handleScrapeSave = async () => {
    if (!trackCode || !raceDate) return;
    setScraping(true);
    setScrapeMessage('');
    try {
      const res = await api.post<ScrapeUpcomingResponse>('/scraping/upcoming', {
        track_code: trackCode,
        race_date: raceDate,
        save: true,
      });
      setScrapePreview(res.data);
      setScrapeMessage(res.data.message ?? '');
    } catch (err: any) {
      setScrapeMessage(err.response?.data?.detail ?? 'Scrape failed.');
    }
    setScraping(false);
  };

  // Pull a scraped race into the manual form for editing/saving
  const importScrapedRace = (race: ScrapedRace) => {
    if (race.race_number) setRaceNumber(String(race.race_number));
    if (race.race_time) setRaceTime(race.race_time);
    if (race.distance_m) setDistanceM(String(race.distance_m));
    if (race.grade) setGrade(race.grade);
    setNumEntries(Math.max(4, Math.min(8, race.entries.length || 6)));
    // setNumEntries triggers the entries-resizing effect; populate after.
    queueMicrotask(() => {
      setEntries(
        race.entries.slice(0, 8).map((e, i) => ({
          trap: e.trap || i + 1,
          dog_name: e.dog_name || '',
          trainer_name: e.trainer_name ?? '',
          weight_kg: e.weight_kg != null ? String(e.weight_kg) : '',
          sp_decimal: '',
        })),
      );
    });
  };

  const validate = (): string | null => {
    if (!trackCode) return 'Pick a track.';
    if (!raceDate) return 'Pick a race date.';
    if (!raceNumber || parseInt(raceNumber) < 1) return 'Race number is required.';
    const dist = parseInt(distanceM);
    if (!dist || dist < 200 || dist > 1000) return 'Distance must be 200–1000 m.';
    const filled = entries.filter((e) => e.dog_name.trim().length > 0);
    if (filled.length < 2) return 'Enter at least 2 dogs.';
    const traps = filled.map((e) => e.trap);
    if (new Set(traps).size !== traps.length) return 'Trap numbers must be unique.';
    return null;
  };

  const handleSave = async () => {
    const err = validate();
    if (err) {
      setSaveError(err);
      return;
    }
    setSaveError('');
    setSaveResult(null);
    setSaving(true);
    try {
      const payload = {
        track_code: trackCode,
        race_date: raceDate,
        race_number: parseInt(raceNumber),
        race_time: raceTime || null,
        distance_m: parseInt(distanceM),
        grade: grade || null,
        going: going || null,
        race_type: 'flat',
        entries: entries
          .filter((e) => e.dog_name.trim().length > 0)
          .map((e) => ({
            trap: e.trap,
            dog_name: e.dog_name.trim(),
            trainer_name: e.trainer_name.trim() || null,
            weight_kg: e.weight_kg ? parseFloat(e.weight_kg) : null,
            sp_decimal: e.sp_decimal ? parseFloat(e.sp_decimal) : null,
          })),
      };
      const res = await api.post<ManualRaceOut>('/races/manual', payload);
      setSaveResult(res.data);
    } catch (err: any) {
      setSaveError(err.response?.data?.detail ?? 'Save failed.');
    }
    setSaving(false);
  };

  const activeTracks = useMemo(() => tracks.filter((t) => t.active), [tracks]);

  return (
    <div>
      <h1 className="text-xl sm:text-2xl font-bold mb-2">Race Card Entry</h1>
      <p className="text-sm text-gray-500 mb-6">
        Scrape an upcoming GRI race card or hand-enter one. Saved races have status
        <span className="font-mono mx-1">scheduled</span>
        and can be predicted from the Predictions page.
      </p>

      {/* Step 1 — Track + date + scrape attempt */}
      <div className="bg-white rounded-lg shadow p-5 mb-6">
        <h2 className="font-semibold mb-3">1. Track & Date</h2>
        <div className="flex gap-3 items-end flex-wrap mb-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Track</label>
            <select
              value={trackCode}
              onChange={(e) => setTrackCode(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm min-w-[180px]"
            >
              <option value="">Select track…</option>
              {activeTracks.map((t) => (
                <option key={t.code} value={t.code}>
                  {t.name} ({t.code})
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Race date</label>
            <input
              type="date"
              value={raceDate}
              onChange={(e) => setRaceDate(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm"
            />
          </div>
          <button
            onClick={handleScrapePreview}
            disabled={scraping || !trackCode}
            className="bg-gray-700 text-white px-4 py-2 rounded-md text-sm hover:bg-gray-800 disabled:opacity-50"
          >
            {scraping ? 'Scraping…' : 'Try Scrape (preview)'}
          </button>
          <button
            onClick={handleScrapeSave}
            disabled={scraping || !trackCode}
            className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {scraping ? 'Scraping…' : 'Scrape & Save All'}
          </button>
        </div>

        {scrapeMessage && (
          <p className="text-sm text-gray-700 bg-gray-50 border rounded p-2">
            {scrapeMessage}
            {scrapePreview?.url_used && (
              <span className="block text-xs text-gray-500 mt-1">
                URL: <span className="font-mono">{scrapePreview.url_used}</span>
              </span>
            )}
          </p>
        )}

        {scrapePreview && scrapePreview.races.length > 0 && (
          <div className="mt-3">
            <p className="text-xs text-gray-500 mb-2">
              Found {scrapePreview.races.length} race(s). Click "Use" to load one
              into the manual form below for review/save.
            </p>
            <div className="flex flex-wrap gap-2">
              {scrapePreview.races.map((r, i) => (
                <button
                  key={i}
                  onClick={() => importScrapedRace(r)}
                  className="text-sm px-3 py-2 rounded-md border bg-white hover:bg-gray-50"
                >
                  R{r.race_number ?? '?'}
                  {r.distance_m ? ` — ${r.distance_m}m` : ''}
                  {r.grade ? ` ${r.grade}` : ''}
                  <span className="text-gray-400 ml-1">
                    ({r.entries.length} dogs)
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Step 2 — Manual race form */}
      <div className="bg-white rounded-lg shadow p-5 mb-6">
        <h2 className="font-semibold mb-3">2. Race Details</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Race #</label>
            <input
              type="number"
              min={1}
              max={20}
              value={raceNumber}
              onChange={(e) => setRaceNumber(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Off time (HH:MM)</label>
            <input
              type="time"
              value={raceTime}
              onChange={(e) => setRaceTime(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Distance (m)</label>
            <input
              type="number"
              min={200}
              max={1000}
              value={distanceM}
              onChange={(e) => setDistanceM(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Grade</label>
            <input
              type="text"
              value={grade}
              onChange={(e) => setGrade(e.target.value)}
              placeholder="A3 / S1 / OR"
              className="border rounded-md px-3 py-2 text-sm w-full"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Going</label>
            <input
              type="text"
              value={going}
              onChange={(e) => setGoing(e.target.value)}
              placeholder="standard"
              className="border rounded-md px-3 py-2 text-sm w-full"
            />
          </div>
        </div>
      </div>

      {/* Step 3 — Entries */}
      <div className="bg-white rounded-lg shadow p-5 mb-6">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <h2 className="font-semibold">3. Runners</h2>
          <div className="flex items-center gap-2 text-sm">
            <label className="text-xs text-gray-500">Traps</label>
            <select
              value={numEntries}
              onChange={(e) => setNumEntries(parseInt(e.target.value))}
              className="border rounded-md px-2 py-1 text-sm"
            >
              {[4, 5, 6, 7, 8].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-[640px]">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-3 py-2 text-left w-16">Trap</th>
                <th className="px-3 py-2 text-left">Dog name</th>
                <th className="px-3 py-2 text-left">Trainer</th>
                <th className="px-3 py-2 text-left w-24">Weight (kg)</th>
                <th className="px-3 py-2 text-left w-24">SP (dec)</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {entries.map((e, i) => (
                <tr key={i}>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      min={1}
                      max={8}
                      value={e.trap}
                      onChange={(ev) => updateEntry(i, 'trap', ev.target.value)}
                      className="border rounded px-2 py-1 w-14 text-center"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="text"
                      value={e.dog_name}
                      onChange={(ev) => updateEntry(i, 'dog_name', ev.target.value)}
                      placeholder="Dog name"
                      className="border rounded px-2 py-1 w-full"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="text"
                      value={e.trainer_name}
                      onChange={(ev) => updateEntry(i, 'trainer_name', ev.target.value)}
                      placeholder="Trainer (optional)"
                      className="border rounded px-2 py-1 w-full"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      step="0.1"
                      value={e.weight_kg}
                      onChange={(ev) => updateEntry(i, 'weight_kg', ev.target.value)}
                      placeholder="32.5"
                      className="border rounded px-2 py-1 w-full"
                    />
                  </td>
                  <td className="px-3 py-2">
                    <input
                      type="number"
                      step="0.01"
                      value={e.sp_decimal}
                      onChange={(ev) => updateEntry(i, 'sp_decimal', ev.target.value)}
                      placeholder="3.50"
                      className="border rounded px-2 py-1 w-full"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Save */}
      <div className="bg-white rounded-lg shadow p-5">
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={handleSave}
            disabled={saving}
            className="bg-green-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-green-700 disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save Race'}
          </button>
          {saveError && <span className="text-sm text-red-600">{saveError}</span>}
        </div>

        {saveResult && (
          <div className="mt-4 bg-green-50 border border-green-200 rounded p-3 text-sm">
            <p className="text-green-800">{saveResult.message}</p>
            <p className="text-green-700 text-xs mt-1">
              Race ID #{saveResult.race_id} —
              {saveResult.dogs_created > 0 && ` ${saveResult.dogs_created} new dog(s) added,`}
              {' '}{saveResult.entries_created} entries.
            </p>
            <div className="mt-3 flex gap-2 flex-wrap">
              <button
                onClick={() => navigate(`/races/${saveResult.race_id}`)}
                className="bg-white border border-green-300 text-green-700 px-3 py-1.5 rounded text-xs hover:bg-green-50"
              >
                View race
              </button>
              <button
                onClick={() => navigate('/predictions')}
                className="bg-green-600 text-white px-3 py-1.5 rounded text-xs hover:bg-green-700"
              >
                Predict it
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
