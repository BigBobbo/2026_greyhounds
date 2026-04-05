import { useEffect, useState } from 'react';
import api from '../api/client';
import type { Track } from '../types/models';

export default function Dashboard() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get<Track[]>('/tracks/').then((res) => {
      setTracks(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow p-5">
          <p className="text-sm text-gray-500">Tracks</p>
          <p className="text-3xl font-bold mt-1">{loading ? '...' : tracks.length}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-5">
          <p className="text-sm text-gray-500">Races</p>
          <p className="text-3xl font-bold mt-1">0</p>
          <p className="text-xs text-gray-400 mt-1">Awaiting first scrape</p>
        </div>
        <div className="bg-white rounded-lg shadow p-5">
          <p className="text-sm text-gray-500">Models</p>
          <p className="text-3xl font-bold mt-1">0</p>
          <p className="text-xs text-gray-400 mt-1">No experiments yet</p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-5">
        <h2 className="text-lg font-semibold mb-3">Irish Tracks</h2>
        {loading ? (
          <p className="text-gray-500">Loading...</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {tracks.map((track) => (
              <div key={track.id} className="border rounded-md p-3">
                <p className="font-medium text-sm">{track.name}</p>
                <p className="text-xs text-gray-500">{track.code} - {track.location}</p>
                <p className="text-xs text-gray-400 mt-1">
                  {track.distances_m?.join('m, ')}m
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
