import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import type { Race } from '../types/models';

export default function RaceList() {
  const [races, setRaces] = useState<Race[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get<Race[]>('/races/').then((res) => {
      setRaces(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Races</h1>

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : races.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 text-lg">No races yet</p>
          <p className="text-gray-400 text-sm mt-1">Run the scraper to populate race data</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-4 py-3">Date</th>
                <th className="px-4 py-3">Track</th>
                <th className="px-4 py-3">Race #</th>
                <th className="px-4 py-3">Distance</th>
                <th className="px-4 py-3">Grade</th>
                <th className="px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {races.map((race) => (
                <tr key={race.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">{race.race_date}</td>
                  <td className="px-4 py-3">
                    <Link to={`/races/${race.id}`} className="text-blue-600 hover:underline">
                      {race.track_name}
                    </Link>
                  </td>
                  <td className="px-4 py-3">{race.race_number}</td>
                  <td className="px-4 py-3">{race.distance_m}m</td>
                  <td className="px-4 py-3">{race.grade || '-'}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs ${
                      race.status === 'resulted' ? 'bg-green-100 text-green-700' :
                      race.status === 'scheduled' ? 'bg-blue-100 text-blue-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {race.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
