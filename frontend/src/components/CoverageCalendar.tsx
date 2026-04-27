import { useEffect, useMemo, useState } from 'react';
import api from '../api/client';
import type { Track } from '../types/models';

interface CalendarDay {
  date: string;
  race_count: number;
  track_count: number;
  tracks: string[];
}

interface CalendarResponse {
  start_date: string;
  end_date: string;
  track_code: string | null;
  days: CalendarDay[];
}

interface Cell {
  date: string;
  day: CalendarDay | null;
  inRange: boolean;
}

const parseISO = (iso: string) => new Date(`${iso}T00:00:00Z`);
const formatISO = (d: Date) => d.toISOString().split('T')[0];

const MONTH_LABELS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

export default function CoverageCalendar() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [trackCode, setTrackCode] = useState<string>('');
  const [data, setData] = useState<CalendarResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [hovered, setHovered] = useState<Cell | null>(null);

  useEffect(() => {
    api.get<Track[]>('/tracks/?active_only=true')
      .then((r) => setTracks(r.data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    const params = trackCode ? { track_code: trackCode } : {};
    api.get<CalendarResponse>('/scraping/coverage-calendar', { params })
      .then((r) => {
        if (cancelled) return;
        setData(r.data);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setData(null);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [trackCode]);

  const requestedTrack = trackCode || null;
  const isStale = data !== null && data.track_code !== requestedTrack;
  const showLoading = loading || isStale;

  const weeks = useMemo<Cell[][]>(() => {
    if (!data) return [];
    const start = parseISO(data.start_date);
    const end = parseISO(data.end_date);

    // Align grid to a Sunday on/before start.
    const gridStart = new Date(start);
    gridStart.setUTCDate(start.getUTCDate() - start.getUTCDay());

    const byDate: Record<string, CalendarDay> = {};
    for (const d of data.days) byDate[d.date] = d;

    const result: Cell[][] = [];
    const cur = new Date(gridStart);
    while (cur.getTime() <= end.getTime()) {
      const week: Cell[] = [];
      for (let i = 0; i < 7; i++) {
        const iso = formatISO(cur);
        week.push({
          date: iso,
          day: byDate[iso] ?? null,
          inRange: cur.getTime() >= start.getTime() && cur.getTime() <= end.getTime(),
        });
        cur.setUTCDate(cur.getUTCDate() + 1);
      }
      result.push(week);
    }
    return result;
  }, [data]);

  const maxValue = useMemo(() => {
    if (!data) return 1;
    const counts = data.days.map((d) => (trackCode ? d.race_count : d.track_count));
    const m = counts.length ? Math.max(...counts) : 0;
    return m || 1;
  }, [data, trackCode]);

  const cellValue = (cell: Cell) => {
    if (!cell.day) return 0;
    return trackCode ? cell.day.race_count : cell.day.track_count;
  };

  const colorFor = (cell: Cell) => {
    if (!cell.inRange) return 'bg-transparent';
    const v = cellValue(cell);
    if (v === 0) return 'bg-gray-100';
    const ratio = v / maxValue;
    if (ratio <= 0.25) return 'bg-emerald-200';
    if (ratio <= 0.5) return 'bg-emerald-400';
    if (ratio <= 0.75) return 'bg-emerald-600';
    return 'bg-emerald-800';
  };

  // Month labels: emit a label when a week's first cell is in a new month.
  const monthLabels = useMemo(() => {
    const labels: { col: number; label: string }[] = [];
    let lastMonth = -1;
    weeks.forEach((week, col) => {
      const firstInRange = week.find((c) => c.inRange);
      if (!firstInRange) return;
      const m = parseISO(firstInRange.date).getUTCMonth();
      if (m !== lastMonth) {
        labels.push({ col, label: MONTH_LABELS[m] });
        lastMonth = m;
      }
    });
    return labels;
  }, [weeks]);

  const totalDaysWithData = data?.days.length ?? 0;
  const totalRaces = data?.days.reduce((acc, d) => acc + d.race_count, 0) ?? 0;

  return (
    <div className="bg-white rounded-lg shadow p-5">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
        <div>
          <h2 className="text-lg font-semibold">Race Coverage</h2>
          <p className="text-xs text-gray-500">
            {data
              ? `${data.start_date} → ${data.end_date} • ${totalDaysWithData} days with races • ${totalRaces.toLocaleString()} races`
              : '—'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-500">Track</label>
          <select
            value={trackCode}
            onChange={(e) => setTrackCode(e.target.value)}
            className="border rounded-md px-2 py-1 text-sm"
          >
            <option value="">All tracks</option>
            {tracks.map((t) => (
              <option key={t.id} value={t.code}>
                {t.code} — {t.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {showLoading ? (
        <p className="text-sm text-gray-400">Loading…</p>
      ) : !data || weeks.length === 0 ? (
        <p className="text-sm text-gray-400">No race data in range.</p>
      ) : (
        <div className="overflow-x-auto">
          <div className="inline-block">
            {/* Month label row */}
            <div
              className="grid mb-1 text-xs text-gray-500"
              style={{
                gridTemplateColumns: `20px repeat(${weeks.length}, 12px)`,
                columnGap: '2px',
              }}
            >
              <div />
              {weeks.map((_, col) => {
                const lbl = monthLabels.find((m) => m.col === col);
                return (
                  <div key={col} className="text-[10px]">
                    {lbl ? lbl.label : ''}
                  </div>
                );
              })}
            </div>

            {/* 7 weekday rows */}
            {[0, 1, 2, 3, 4, 5, 6].map((row) => (
              <div
                key={row}
                className="grid mb-[2px]"
                style={{
                  gridTemplateColumns: `20px repeat(${weeks.length}, 12px)`,
                  columnGap: '2px',
                }}
              >
                <div className="text-[10px] text-gray-400 leading-[12px]">
                  {row === 1 ? 'Mon' : row === 3 ? 'Wed' : row === 5 ? 'Fri' : ''}
                </div>
                {weeks.map((week, col) => {
                  const cell = week[row];
                  return (
                    <div
                      key={col}
                      className={`w-3 h-3 rounded-sm ${colorFor(cell)}`}
                      onMouseEnter={() => cell.inRange && setHovered(cell)}
                      onMouseLeave={() => setHovered(null)}
                      title={
                        cell.inRange
                          ? cell.day
                            ? trackCode
                              ? `${cell.date}: ${cell.day.race_count} races`
                              : `${cell.date}: ${cell.day.track_count} track(s), ${cell.day.race_count} races`
                            : `${cell.date}: no races`
                          : ''
                      }
                    />
                  );
                })}
              </div>
            ))}

            {/* Legend */}
            <div className="flex items-center gap-2 mt-3 text-[11px] text-gray-500">
              <span>Less</span>
              <span className="w-3 h-3 rounded-sm bg-gray-100" />
              <span className="w-3 h-3 rounded-sm bg-emerald-200" />
              <span className="w-3 h-3 rounded-sm bg-emerald-400" />
              <span className="w-3 h-3 rounded-sm bg-emerald-600" />
              <span className="w-3 h-3 rounded-sm bg-emerald-800" />
              <span>More</span>
              {hovered && hovered.day && (
                <span className="ml-4 text-gray-700">
                  {hovered.date}:{' '}
                  {trackCode
                    ? `${hovered.day.race_count} races`
                    : `${hovered.day.track_count} tracks (${hovered.day.tracks.join(', ')}), ${hovered.day.race_count} races`}
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
