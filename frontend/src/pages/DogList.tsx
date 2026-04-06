import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../api/client';
import type { Dog } from '../types/models';

export default function DogList() {
  const [dogs, setDogs] = useState<Dog[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchDogs = (query: string) => {
    setLoading(true);
    const params: Record<string, string> = {};
    if (query) params.search = query;
    api.get<Dog[]>('/dogs/', { params }).then((res) => {
      setDogs(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  };

  useEffect(() => { fetchDogs(''); }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchDogs(search);
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Dogs</h1>

      <form onSubmit={handleSearch} className="bg-white rounded-lg shadow p-4 mb-4 flex gap-3 items-end">
        <div className="flex-1 max-w-md">
          <label className="block text-xs text-gray-500 mb-1">Search by name</label>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="e.g. DROOPYS, BALLYMAC..."
            className="border rounded-md px-3 py-2 text-sm w-full"
          />
        </div>
        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700">
          Search
        </button>
        {search && (
          <button type="button" onClick={() => { setSearch(''); fetchDogs(''); }} className="text-gray-500 text-sm hover:text-gray-700">
            Clear
          </button>
        )}
      </form>

      {loading ? (
        <div className="flex items-center justify-center h-32"><p className="text-gray-400">Searching...</p></div>
      ) : dogs.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 text-lg">No dogs found</p>
          <p className="text-gray-400 text-sm mt-1">{search ? 'Try a different search' : 'Run the scraper to populate dog data'}</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Trainer</th>
                <th className="px-4 py-3">Sire</th>
                <th className="px-4 py-3">Dam</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {dogs.map((dog) => (
                <tr key={dog.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link to={`/dogs/${dog.id}`} className="text-blue-600 hover:underline font-medium">
                      {dog.name}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-gray-600">{dog.trainer_name || '-'}</td>
                  <td className="px-4 py-3 text-gray-500 text-xs">{dog.sire || '-'}</td>
                  <td className="px-4 py-3 text-gray-500 text-xs">{dog.dam || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="px-4 py-3 text-xs text-gray-400 bg-gray-50">
            Showing {dogs.length} dogs
          </div>
        </div>
      )}
    </div>
  );
}
