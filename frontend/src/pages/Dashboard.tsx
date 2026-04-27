import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import CoverageCalendar from '../components/CoverageCalendar';

interface ScrapingStatus {
  total_races: number;
  total_entries: number;
  total_dogs: number;
  total_tracks: number;
  last_scrape: { status: string; started_at: string | null } | null;
}

interface Experiment {
  id: number;
  name: string;
  algorithm: string;
  target: string;
  status: string;
  metrics: Record<string, number> | null;
}

export default function Dashboard() {
  const [stats, setStats] = useState<ScrapingStatus | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api.get<ScrapingStatus>('/scraping/status'),
      api.get<Experiment[]>('/training/experiments?limit=5'),
    ]).then(([statsRes, expRes]) => {
      setStats(statsRes.data);
      setExperiments(expRes.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="text-xl sm:text-2xl font-bold mb-4 sm:mb-6">Dashboard</h1>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow p-5">
          <p className="text-sm text-gray-500">Races</p>
          <p className="text-3xl font-bold mt-1">{loading ? '...' : stats?.total_races?.toLocaleString()}</p>
          <Link to="/races" className="text-xs text-blue-500 hover:underline">View all</Link>
        </div>
        <div className="bg-white rounded-lg shadow p-5">
          <p className="text-sm text-gray-500">Dogs</p>
          <p className="text-3xl font-bold mt-1">{loading ? '...' : stats?.total_dogs?.toLocaleString()}</p>
          <Link to="/dogs" className="text-xs text-blue-500 hover:underline">Search dogs</Link>
        </div>
        <div className="bg-white rounded-lg shadow p-5">
          <p className="text-sm text-gray-500">Race Entries</p>
          <p className="text-3xl font-bold mt-1">{loading ? '...' : stats?.total_entries?.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-5">
          <p className="text-sm text-gray-500">Tracks</p>
          <p className="text-3xl font-bold mt-1">{loading ? '...' : stats?.total_tracks}</p>
        </div>
      </div>

      {/* Coverage calendar */}
      <div className="mb-6">
        <CoverageCalendar />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Scraping status */}
        <div className="bg-white rounded-lg shadow p-5">
          <h2 className="text-lg font-semibold mb-3">Scraping</h2>
          {stats?.last_scrape ? (
            <div>
              <p className="text-sm">
                Status: <span className={`font-medium ${
                  stats.last_scrape.status === 'running' ? 'text-yellow-600' :
                  stats.last_scrape.status === 'success' ? 'text-green-600' : 'text-gray-600'
                }`}>{stats.last_scrape.status}</span>
              </p>
              {stats.last_scrape.started_at && (
                <p className="text-xs text-gray-400 mt-1">
                  Started: {new Date(stats.last_scrape.started_at).toLocaleString()}
                </p>
              )}
            </div>
          ) : (
            <p className="text-gray-400 text-sm">No scrapes yet</p>
          )}
          <Link to="/scraping" className="text-xs text-blue-500 hover:underline mt-2 inline-block">
            Manage scraping
          </Link>
        </div>

        {/* Recent experiments */}
        <div className="bg-white rounded-lg shadow p-5">
          <h2 className="text-lg font-semibold mb-3">Recent Experiments</h2>
          {experiments.length === 0 ? (
            <p className="text-gray-400 text-sm">No experiments yet</p>
          ) : (
            <ul className="space-y-2">
              {experiments.map(exp => (
                <li key={exp.id} className="flex items-center justify-between text-sm">
                  <Link to={`/training/${exp.id}`} className="text-blue-600 hover:underline">
                    {exp.name}
                  </Link>
                  <span className={`px-2 py-0.5 rounded-full text-xs ${
                    exp.status === 'completed' ? 'bg-green-100 text-green-700' :
                    exp.status === 'running' ? 'bg-yellow-100 text-yellow-700' :
                    exp.status === 'failed' ? 'bg-red-100 text-red-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {exp.status}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <Link to="/training" className="text-xs text-blue-500 hover:underline mt-2 inline-block">
            Training Lab
          </Link>
        </div>
      </div>
    </div>
  );
}
