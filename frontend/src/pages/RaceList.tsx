import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import type { Race, Track } from '../types/models';

export default function RaceList() {
  const [races, setRaces] = useState<Race[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(true);
  const [trackFilter, setTrackFilter] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  const fetchRaces = () => {
    setLoading(true);
    const params: Record<string, string> = { limit: '100' };
    if (trackFilter) params.track_id = trackFilter;
    if (dateFrom) params.date_from = dateFrom;
    if (dateTo) params.date_to = dateTo;

    api.get<Race[]>('/races/', { params }).then(res => {
      setRaces(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  };

  // Initial load: state updates happen inside the promise chains (loading
  // starts true) so no setState runs synchronously in the effect body. No
  // filters are set on mount, so this matches fetchRaces() with defaults.
  useEffect(() => {
    api.get<Track[]>('/tracks/').then(res => setTracks(res.data));
    api.get<Race[]>('/races/', { params: { limit: '100' } }).then(res => {
      setRaces(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const handleFilter = (e: React.FormEvent) => {
    e.preventDefault();
    fetchRaces();
  };

  return (
    <div>
      <h1 className="text-xl sm:text-2xl font-bold mb-4">Races</h1>

      {/* Filters */}
      <form onSubmit={handleFilter} className="bg-white rounded-lg shadow p-4 mb-4 flex flex-wrap gap-3 items-end">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Track</label>
          <select
            value={trackFilter}
            onChange={e => setTrackFilter(e.target.value)}
            className="border rounded-md px-3 py-2 text-sm"
          >
            <option value="">All tracks</option>
            {tracks.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">From</label>
          <input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} className="border rounded-md px-3 py-2 text-sm" />
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">To</label>
          <input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} className="border rounded-md px-3 py-2 text-sm" />
        </div>
        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700">
          Filter
        </button>
        <button type="button" onClick={() => { setTrackFilter(''); setDateFrom(''); setDateTo(''); setTimeout(fetchRaces, 0); }} className="text-gray-500 text-sm hover:text-gray-700">
          Clear
        </button>
      </form>

      {loading ? (
        <div className="flex items-center justify-center h-32"><p className="text-gray-400">Loading races...</p></div>
      ) : races.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 text-lg">No races found</p>
          <p className="text-gray-400 text-sm mt-1">Try adjusting your filters or run the scraper</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden overflow-x-auto">
          <table className="w-full text-sm text-left min-w-[600px]">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-3 sm:px-4 py-3">Date</th>
                <th className="px-3 sm:px-4 py-3">Track</th>
                <th className="px-3 sm:px-4 py-3">Race #</th>
                <th className="px-3 sm:px-4 py-3">Distance</th>
                <th className="px-3 sm:px-4 py-3">Grade</th>
                <th className="px-3 sm:px-4 py-3">Runners</th>
                <th className="px-3 sm:px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {races.map((race) => (
                <tr key={race.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-xs">{race.race_date}</td>
                  <td className="px-4 py-3">
                    <Link to={`/races/${race.id}`} className="text-blue-600 hover:underline font-medium">
                      {race.track_name}
                    </Link>
                  </td>
                  <td className="px-4 py-3">{race.race_number}</td>
                  <td className="px-4 py-3">{race.distance_m}m</td>
                  <td className="px-4 py-3">{race.grade || '-'}</td>
                  <td className="px-4 py-3">{race.num_runners || '-'}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs ${
                      race.status === 'resulted' ? 'bg-green-100 text-green-700' :
                      race.status === 'scheduled' ? 'bg-blue-100 text-blue-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>{race.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="px-4 py-3 text-xs text-gray-400 bg-gray-50">
            Showing {races.length} races
          </div>
        </div>
      )}
    </div>
  );
}
