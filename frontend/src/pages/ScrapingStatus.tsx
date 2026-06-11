import { useEffect, useMemo, useState } from 'react';
import api from '../api/client';
import type { Track } from '../types/models';

interface ScrapeLog {
  id: number;
  spider_name: string;
  source: string | null;
  status: string;
  records_scraped: number;
  records_new: number;
  records_updated: number;
  error_message: string | null;
  started_at: string | null;
  heartbeat_at: string | null;
  completed_at: string | null;
}

interface ScrapingStatus {
  total_races: number;
  total_entries: number;
  total_dogs: number;
  total_tracks: number;
  last_scrape: ScrapeLog | null;
  recent_logs: ScrapeLog[];
}

interface LastScrapeInfo {
  last_race_date: string | null;
  proposed_start_date: string | null;
  today: string;
  days_to_scrape: number;
  active_track_count: number;
}

const isoDaysAgo = (days: number) => {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().split('T')[0];
};

export default function ScrapingStatusPage() {
  const [status, setStatus] = useState<ScrapingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [triggerTrack, setTriggerTrack] = useState('SHP');
  const [triggerDate, setTriggerDate] = useState(new Date().toISOString().split('T')[0]);
  const [triggering, setTriggering] = useState(false);
  const [message, setMessage] = useState('');
  const [lastInfo, setLastInfo] = useState<LastScrapeInfo | null>(null);
  const [scrapingSinceLast, setScrapingSinceLast] = useState(false);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [rangeStart, setRangeStart] = useState(isoDaysAgo(30));
  const [rangeEnd, setRangeEnd] = useState(new Date().toISOString().split('T')[0]);
  const [rangeTracks, setRangeTracks] = useState<string[]>([]);
  const [rangeAllActive, setRangeAllActive] = useState(true);
  const [rangeRunning, setRangeRunning] = useState(false);
  const [reaping, setReaping] = useState(false);
  // "Now" is sampled once per poll tick (see fetchStatus) instead of during
  // render, so heartbeat staleness stays pure but refreshes with each poll.
  const [now, setNow] = useState(() => Date.now());

  const fetchStatus = () => {
    api.get<ScrapingStatus>('/scraping/status').then((res) => {
      setNow(Date.now());
      setStatus(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
    api.get<LastScrapeInfo>('/scraping/last-scrape-info').then((res) => {
      setLastInfo(res.data);
    }).catch(() => {});
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    api.get<Track[]>('/tracks/?active_only=true')
      .then((res) => setTracks(res.data))
      .catch(() => {});
  }, []);

  const dayCount = useMemo(() => {
    if (!rangeStart || !rangeEnd) return 0;
    const a = new Date(rangeStart);
    const b = new Date(rangeEnd);
    if (Number.isNaN(a.getTime()) || Number.isNaN(b.getTime()) || b < a) return 0;
    return Math.round((b.getTime() - a.getTime()) / 86400000) + 1;
  }, [rangeStart, rangeEnd]);

  const handleRangeScrape = async () => {
    setRangeRunning(true);
    setMessage('');
    try {
      const res = await api.post('/scraping/backfill', {
        start_date: rangeStart,
        end_date: rangeEnd,
        track_codes: rangeAllActive ? null : rangeTracks,
      });
      setMessage(res.data.message);
      setTimeout(fetchStatus, 2000);
    } catch {
      setMessage('Failed to start date-range scrape');
    }
    setRangeRunning(false);
  };

  const toggleRangeTrack = (code: string) => {
    setRangeTracks((cur) =>
      cur.includes(code) ? cur.filter((c) => c !== code) : [...cur, code]
    );
  };

  const handleScrapeSinceLast = async () => {
    setScrapingSinceLast(true);
    setMessage('');
    try {
      const res = await api.post('/scraping/scrape-since-last-race-date', {});
      setMessage(res.data.message);
      setTimeout(fetchStatus, 2000);
    } catch {
      setMessage('Failed to start scrape since last race date');
    }
    setScrapingSinceLast(false);
  };

  const handleTrigger = async () => {
    setTriggering(true);
    setMessage('');
    try {
      const res = await api.post('/scraping/trigger', {
        track_code: triggerTrack,
        date_from: triggerDate,
      });
      setMessage(res.data.message);
      setTimeout(fetchStatus, 2000);
    } catch {
      setMessage('Failed to trigger scrape');
    }
    setTriggering(false);
  };

  const handleDiscoverTracks = async () => {
    try {
      await api.post('/scraping/discover-tracks');
      setMessage('Track discovery started in background');
    } catch {
      setMessage('Failed to start track discovery');
    }
  };

  const handleReapStale = async () => {
    setReaping(true);
    setMessage('');
    try {
      const res = await api.post<{ reaped: number; log_ids: number[] }>(
        '/scraping/reap-stale',
        { stale_minutes: 15 }
      );
      if (res.data.reaped === 0) {
        setMessage('No stale running jobs found.');
      } else {
        setMessage(
          `Marked ${res.data.reaped} stuck job${res.data.reaped === 1 ? '' : 's'} as failed (ids: ${res.data.log_ids.join(', ')}).`
        );
      }
      setTimeout(fetchStatus, 500);
    } catch {
      setMessage('Failed to reap stale jobs');
    }
    setReaping(false);
  };

  const statusColor = (s: string) => {
    switch (s) {
      case 'success': return 'bg-green-100 text-green-700';
      case 'running': return 'bg-yellow-100 text-yellow-700';
      case 'failed': return 'bg-red-100 text-red-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div>
      <h1 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6">Scraping Status</h1>

      {/* Stats cards */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Races</p>
            <p className="text-2xl font-bold">{status.total_races}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Race Entries</p>
            <p className="text-2xl font-bold">{status.total_entries}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Dogs</p>
            <p className="text-2xl font-bold">{status.total_dogs}</p>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <p className="text-sm text-gray-500">Active Tracks</p>
            <p className="text-2xl font-bold">{status.total_tracks}</p>
          </div>
        </div>
      )}

      {/* Scrape since last race date */}
      <div className="bg-white rounded-lg shadow p-5 mb-6">
        <h2 className="font-semibold mb-3">Scrape Since Last Race Date</h2>
        <p className="text-xs text-gray-500 mb-3">
          Picks up from the day after the most recent race already in the database
          (not the timestamp of the last scrape job).
        </p>
        {lastInfo ? (
          lastInfo.last_race_date ? (
            <div className="text-sm text-gray-700 mb-3 space-y-1">
              <p>
                Latest race date in DB:{' '}
                <span className="font-mono font-semibold">{lastInfo.last_race_date}</span>
              </p>
              {lastInfo.days_to_scrape > 0 ? (
                <p>
                  Will scrape{' '}
                  <span className="font-mono">{lastInfo.proposed_start_date}</span> →{' '}
                  <span className="font-mono">{lastInfo.today}</span>{' '}
                  ({lastInfo.days_to_scrape} day{lastInfo.days_to_scrape === 1 ? '' : 's'} ×{' '}
                  {lastInfo.active_track_count} active tracks)
                </p>
              ) : (
                <p className="text-green-700">Already up to date.</p>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500 mb-3">
              No prior scraped races found. Use a backfill with explicit dates first.
            </p>
          )
        ) : (
          <p className="text-sm text-gray-400 mb-3">Loading…</p>
        )}
        <button
          onClick={handleScrapeSinceLast}
          disabled={
            scrapingSinceLast ||
            !lastInfo?.last_race_date ||
            (lastInfo?.days_to_scrape ?? 0) === 0
          }
          className="bg-indigo-600 text-white px-4 py-2 rounded-md text-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {scrapingSinceLast ? 'Starting…' : 'Scrape Since Last Race Date'}
        </button>
      </div>

      {/* Date range scrape across tracks */}
      <div className="bg-white rounded-lg shadow p-5 mb-6">
        <h2 className="font-semibold mb-3">Date Range Scrape</h2>
        <p className="text-xs text-gray-500 mb-3">
          Scrape every day between two dates across selected tracks. Existing
          races (matched by track + date + race number) are updated, never
          duplicated.
        </p>
        <div className="flex gap-3 items-end flex-wrap mb-3">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Start Date</label>
            <input
              type="date"
              value={rangeStart}
              onChange={(e) => setRangeStart(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">End Date</label>
            <input
              type="date"
              value={rangeEnd}
              onChange={(e) => setRangeEnd(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm"
            />
          </div>
          <label className="flex items-center gap-2 text-sm pb-2">
            <input
              type="checkbox"
              checked={rangeAllActive}
              onChange={(e) => setRangeAllActive(e.target.checked)}
            />
            All active tracks
          </label>
        </div>

        {!rangeAllActive && (
          <div className="mb-3">
            <p className="text-xs text-gray-500 mb-2">
              Pick tracks ({rangeTracks.length} selected):
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 max-h-44 overflow-y-auto border rounded-md p-2">
              {tracks.map((t) => (
                <label
                  key={t.id}
                  className="flex items-center gap-2 text-xs cursor-pointer hover:bg-gray-50 rounded px-1 py-0.5"
                >
                  <input
                    type="checkbox"
                    checked={rangeTracks.includes(t.code)}
                    onChange={() => toggleRangeTrack(t.code)}
                  />
                  <span className="font-mono">{t.code}</span>
                  <span className="text-gray-500 truncate">{t.name}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={handleRangeScrape}
            disabled={
              rangeRunning ||
              dayCount === 0 ||
              (!rangeAllActive && rangeTracks.length === 0)
            }
            className="bg-purple-600 text-white px-4 py-2 rounded-md text-sm hover:bg-purple-700 disabled:opacity-50"
          >
            {rangeRunning ? 'Starting…' : 'Scrape Date Range'}
          </button>
          {dayCount > 0 && (
            <span className="text-xs text-gray-500">
              {dayCount} day{dayCount === 1 ? '' : 's'} ×{' '}
              {rangeAllActive
                ? `${status?.total_tracks ?? '?'} active tracks`
                : `${rangeTracks.length} track${rangeTracks.length === 1 ? '' : 's'}`}
            </span>
          )}
        </div>
      </div>

      {/* Manual trigger */}
      <div className="bg-white rounded-lg shadow p-5 mb-6">
        <h2 className="font-semibold mb-3">Manual Scrape</h2>
        <div className="flex gap-3 items-end flex-wrap">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Track Code</label>
            <input
              type="text"
              value={triggerTrack}
              onChange={(e) => setTriggerTrack(e.target.value.toUpperCase())}
              className="border rounded-md px-3 py-2 text-sm w-24"
              placeholder="SHP"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Date</label>
            <input
              type="date"
              value={triggerDate}
              onChange={(e) => setTriggerDate(e.target.value)}
              className="border rounded-md px-3 py-2 text-sm"
            />
          </div>
          <button
            onClick={handleTrigger}
            disabled={triggering}
            className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {triggering ? 'Starting...' : 'Scrape'}
          </button>
          <button
            onClick={handleDiscoverTracks}
            className="bg-gray-600 text-white px-4 py-2 rounded-md text-sm hover:bg-gray-700"
          >
            Discover Tracks
          </button>
        </div>
        {message && (
          <p className="mt-2 text-sm text-blue-600">{message}</p>
        )}
      </div>

      {/* Recent scrape logs */}
      <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
        <div className="flex items-center justify-between px-5 pt-4 pb-2 gap-3 flex-wrap">
          <h2 className="font-semibold">Recent Scrape Logs</h2>
          <button
            onClick={handleReapStale}
            disabled={reaping}
            className="text-xs bg-red-600 text-white px-3 py-1.5 rounded-md hover:bg-red-700 disabled:opacity-50"
            title="Mark any 'running' jobs without a recent heartbeat as failed"
          >
            {reaping ? 'Reaping…' : 'Reap stale jobs'}
          </button>
        </div>
        {loading ? (
          <p className="px-5 py-4 text-gray-500">Loading...</p>
        ) : !status?.recent_logs.length ? (
          <p className="px-5 py-4 text-gray-400">No scrapes have been run yet</p>
        ) : (
          <table className="w-full text-sm text-left min-w-[640px]">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-3 sm:px-4 py-3">Spider</th>
                <th className="px-3 sm:px-4 py-3">Source</th>
                <th className="px-3 sm:px-4 py-3">Status</th>
                <th className="px-3 sm:px-4 py-3">Records</th>
                <th className="px-3 sm:px-4 py-3">New</th>
                <th className="px-3 sm:px-4 py-3">Started</th>
                <th className="px-3 sm:px-4 py-3">Duration</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {status.recent_logs.map((log) => {
                const duration = log.started_at && log.completed_at
                  ? Math.round((new Date(log.completed_at).getTime() - new Date(log.started_at).getTime()) / 1000)
                  : null;
                const isRunning = log.status === 'running';
                const lastAlive = log.heartbeat_at || log.started_at;
                const minsSinceAlive = isRunning && lastAlive
                  ? Math.round((now - new Date(lastAlive).getTime()) / 60000)
                  : null;
                const isStale = isRunning && minsSinceAlive !== null && minsSinceAlive >= 15;
                return (
                  <tr key={log.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3">{log.spider_name}</td>
                    <td
                      className="px-4 py-3 text-gray-500 text-xs max-w-[320px] truncate"
                      title={log.source ?? ''}
                    >
                      {log.source}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-xs ${statusColor(log.status)}`}>
                        {log.status}
                      </span>
                      {isStale && (
                        <span
                          className="ml-1 px-2 py-0.5 rounded-full text-xs bg-red-100 text-red-700"
                          title={`No heartbeat for ${minsSinceAlive} minutes — likely crashed`}
                        >
                          stale
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">{log.records_scraped}</td>
                    <td className="px-4 py-3">{log.records_new}</td>
                    <td className="px-4 py-3 text-xs text-gray-500">
                      {log.started_at ? new Date(log.started_at).toLocaleString() : '-'}
                    </td>
                    <td className="px-4 py-3 text-xs">
                      {duration !== null
                        ? `${duration}s`
                        : isRunning
                          ? minsSinceAlive !== null
                            ? `running (${minsSinceAlive}m since heartbeat)`
                            : 'running...'
                          : '-'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
