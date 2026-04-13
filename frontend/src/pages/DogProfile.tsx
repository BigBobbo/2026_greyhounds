import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../api/client';
import type { Dog } from '../types/models';

interface FormEntry {
  race_id: number;
  race_date: string;
  race_number: number;
  track_name: string;
  track_code: string;
  distance_m: number;
  grade: string;
  going: string | null;
  trap: number | null;
  finish_position: number | null;
  finish_time: number | null;
  beaten_distance: number | null;
  weight_kg: number | null;
  starting_price: string | null;
  sp_decimal: number | null;
  comment: string | null;
}

interface DogStats {
  total_runs: number;
  wins: number;
  places: number;
  win_pct: number;
  place_pct: number;
  avg_time: number | null;
  best_time: number | null;
}

interface DogFormData {
  dog: Dog;
  stats: DogStats;
  form: FormEntry[];
}

const TRAP_COLORS: Record<number, string> = {
  1: 'bg-red-500 text-white', 2: 'bg-blue-500 text-white',
  3: 'bg-white border border-gray-300 text-gray-900', 4: 'bg-green-500 text-white',
  5: 'bg-orange-400 text-white', 6: 'bg-black text-white',
};

export default function DogProfile() {
  const { id } = useParams();
  const [data, setData] = useState<DogFormData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get<DogFormData>(`/dogs/${id}/form`).then(res => {
      setData(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="flex items-center justify-center h-64"><p className="text-gray-400">Loading...</p></div>;
  if (!data) return <p className="text-red-500">Dog not found</p>;

  const { dog, stats, form } = data;

  // Chart data — chronological
  const chartForm = [...form].reverse();
  const timeData = chartForm
    .filter(f => f.finish_time)
    .map((f, i) => ({ run: i + 1, time: f.finish_time, date: f.race_date }));
  const posData = chartForm
    .filter(f => f.finish_position)
    .map((f, i) => ({ run: i + 1, position: f.finish_position, date: f.race_date }));

  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <Link to="/dogs" className="text-gray-400 hover:text-gray-600 text-lg">&larr;</Link>
        <h1 className="text-xl sm:text-2xl font-bold">{dog.name}</h1>
      </div>

      <p className="text-gray-500 mb-6">
        {dog.trainer_name && <>Trainer: {dog.trainer_name} &middot; </>}
        {dog.sire && <>Sire: {dog.sire} &middot; </>}
        {dog.dam && <>Dam: {dog.dam}</>}
      </p>

      {/* Stats cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-3 mb-6">
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Runs</p>
          <p className="text-2xl font-bold">{stats.total_runs}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Wins</p>
          <p className="text-2xl font-bold text-green-600">{stats.wins}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Places</p>
          <p className="text-2xl font-bold">{stats.places}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Win %</p>
          <p className="text-2xl font-bold">{stats.win_pct}%</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Place %</p>
          <p className="text-2xl font-bold">{stats.place_pct}%</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Avg Time</p>
          <p className="text-2xl font-bold">{stats.avg_time || '-'}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-4">
          <p className="text-xs text-gray-500">Best Time</p>
          <p className="text-2xl font-bold text-blue-600">{stats.best_time || '-'}</p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {timeData.length > 2 && (
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="font-semibold mb-3">Finish Time Trend</h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={timeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="run" label={{ value: 'Run #', position: 'bottom', offset: -5 }} />
                <YAxis domain={['auto', 'auto']} label={{ value: 'Time (s)', angle: -90, position: 'left' }} />
                <Tooltip labelFormatter={(v) => `Run ${v}`} formatter={(v: any) => [Number(v).toFixed(2) + 's', 'Time']} />
                <Line type="monotone" dataKey="time" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {posData.length > 2 && (
          <div className="bg-white rounded-lg shadow p-5">
            <h2 className="font-semibold mb-3">Finish Position Trend</h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={posData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="run" label={{ value: 'Run #', position: 'bottom', offset: -5 }} />
                <YAxis reversed domain={[1, 6]} label={{ value: 'Position', angle: -90, position: 'left' }} />
                <Tooltip labelFormatter={(v) => `Run ${v}`} formatter={(v: any) => [v, 'Position']} />
                <Line type="monotone" dataKey="position" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Form table */}
      <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
        <h2 className="font-semibold px-5 pt-4 pb-2">Race Form ({form.length} runs)</h2>
        <table className="w-full text-sm text-left min-w-[800px]">
          <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
            <tr>
              <th className="px-3 sm:px-4 py-3">Date</th>
              <th className="px-3 sm:px-4 py-3">Track</th>
              <th className="px-3 sm:px-4 py-3">Dist</th>
              <th className="px-3 sm:px-4 py-3">Grade</th>
              <th className="px-3 sm:px-4 py-3">Trap</th>
              <th className="px-3 sm:px-4 py-3">Pos</th>
              <th className="px-3 sm:px-4 py-3">Time</th>
              <th className="px-3 sm:px-4 py-3">Btn</th>
              <th className="px-3 sm:px-4 py-3">Wt</th>
              <th className="px-3 sm:px-4 py-3">SP</th>
              <th className="px-3 sm:px-4 py-3">Comment</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {form.map((f, i) => (
              <tr key={i} className={f.finish_position === 1 ? 'bg-yellow-50' : 'hover:bg-gray-50'}>
                <td className="px-4 py-2 text-xs">{f.race_date}</td>
                <td className="px-4 py-2">
                  <Link to={`/races/${f.race_id}`} className="text-blue-600 hover:underline text-xs">
                    {f.track_name}
                  </Link>
                </td>
                <td className="px-4 py-2 text-xs">{f.distance_m}m</td>
                <td className="px-4 py-2 text-xs">{f.grade || '-'}</td>
                <td className="px-4 py-2">
                  {f.trap ? (
                    <span className={`inline-flex items-center justify-center w-6 h-6 rounded-sm text-xs font-bold ${TRAP_COLORS[f.trap] || 'bg-gray-300 text-white'}`}>
                      {f.trap}
                    </span>
                  ) : '-'}
                </td>
                <td className="px-4 py-2">
                  <span className={f.finish_position === 1 ? 'font-bold text-yellow-700' : f.finish_position && f.finish_position <= 3 ? 'font-medium' : 'text-gray-500'}>
                    {f.finish_position || '-'}
                  </span>
                </td>
                <td className="px-4 py-2 font-mono text-xs">{f.finish_time?.toFixed(2) || '-'}</td>
                <td className="px-4 py-2 text-xs text-gray-500">{f.beaten_distance ? `${f.beaten_distance}L` : '-'}</td>
                <td className="px-4 py-2 text-xs">{f.weight_kg || '-'}</td>
                <td className="px-4 py-2 font-mono text-xs">{f.starting_price || '-'}</td>
                <td className="px-4 py-2 text-xs text-gray-400 max-w-[150px] truncate">{f.comment || ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
