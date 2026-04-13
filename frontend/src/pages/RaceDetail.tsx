import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../api/client';
import type { RaceDetail as RaceDetailType } from '../types/models';

const TRAP_COLORS: Record<number, string> = {
  1: 'bg-red-500',
  2: 'bg-blue-500',
  3: 'bg-white border border-gray-300',
  4: 'bg-green-500',
  5: 'bg-orange-400',
  6: 'bg-black',
};

const TRAP_TEXT: Record<number, string> = {
  1: 'text-white', 2: 'text-white', 3: 'text-gray-900',
  4: 'text-white', 5: 'text-white', 6: 'text-white',
};

export default function RaceDetail() {
  const { id } = useParams();
  const [race, setRace] = useState<RaceDetailType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get<RaceDetailType>(`/races/${id}`).then(res => {
      setRace(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="flex items-center justify-center h-64"><p className="text-gray-400">Loading race...</p></div>;
  if (!race) return <p className="text-red-500">Race not found</p>;

  const statusColor = race.status === 'resulted' ? 'bg-green-100 text-green-700' :
    race.status === 'scheduled' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700';

  const entries = [...(race.entries || [])].sort((a, b) => {
    if (a.finish_position && b.finish_position) return a.finish_position - b.finish_position;
    if (a.finish_position) return -1;
    if (b.finish_position) return 1;
    return (a.trap || 0) - (b.trap || 0);
  });

  return (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <Link to="/races" className="text-gray-400 hover:text-gray-600 text-lg">&larr;</Link>
        <h1 className="text-xl sm:text-2xl font-bold">{race.track_name}</h1>
        <span className={`px-2 py-0.5 rounded-full text-xs ${statusColor}`}>{race.status}</span>
      </div>

      <p className="text-gray-500 mb-6">
        Race {race.race_number} &middot; {race.race_date} &middot; {race.distance_m}m &middot; Grade {race.grade || '-'}
        {race.going && <> &middot; Going: {race.going}</>}
      </p>

      <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
        <table className="w-full text-sm text-left min-w-[640px]">
          <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
            <tr>
              <th className="px-3 sm:px-4 py-3 w-16">Pos</th>
              <th className="px-3 sm:px-4 py-3 w-16">Trap</th>
              <th className="px-3 sm:px-4 py-3">Dog</th>
              <th className="px-3 sm:px-4 py-3">Time</th>
              <th className="px-3 sm:px-4 py-3">Beaten</th>
              <th className="px-3 sm:px-4 py-3">Weight</th>
              <th className="px-3 sm:px-4 py-3">SP</th>
              <th className="px-3 sm:px-4 py-3">Comment</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {entries.map((entry) => (
              <tr key={entry.id} className={entry.finish_position === 1 ? 'bg-yellow-50' : 'hover:bg-gray-50'}>
                <td className="px-4 py-3">
                  <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-sm font-bold ${
                    entry.finish_position === 1 ? 'bg-yellow-400 text-yellow-900' :
                    entry.finish_position && entry.finish_position <= 3 ? 'bg-gray-200 text-gray-700' :
                    'text-gray-500'
                  }`}>
                    {entry.finish_position || '-'}
                  </span>
                </td>
                <td className="px-4 py-3">
                  {entry.trap ? (
                    <span className={`inline-flex items-center justify-center w-7 h-7 rounded-sm text-xs font-bold ${TRAP_COLORS[entry.trap] || 'bg-gray-300'} ${TRAP_TEXT[entry.trap] || 'text-white'}`}>
                      {entry.trap}
                    </span>
                  ) : '-'}
                </td>
                <td className="px-4 py-3">
                  <Link to={`/dogs/${entry.dog_id}`} className="text-blue-600 hover:underline font-medium">
                    {entry.dog_name || `Dog #${entry.dog_id}`}
                  </Link>
                </td>
                <td className="px-4 py-3 font-mono">
                  {entry.finish_time ? entry.finish_time.toFixed(2) : '-'}
                </td>
                <td className="px-4 py-3 font-mono text-gray-500">
                  {entry.beaten_distance ? `${entry.beaten_distance}L` : entry.finish_position === 1 ? '-' : ''}
                </td>
                <td className="px-4 py-3 text-gray-500">
                  {entry.weight_kg || '-'}
                </td>
                <td className="px-4 py-3 font-mono">
                  {entry.starting_price || '-'}
                  {entry.sp_decimal && <span className="text-gray-400 text-xs ml-1">({entry.sp_decimal})</span>}
                </td>
                <td className="px-4 py-3 text-xs text-gray-500 max-w-[200px] truncate">
                  {entry.comment || '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
