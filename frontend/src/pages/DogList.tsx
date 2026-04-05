import { useEffect, useState } from 'react';
import api from '../api/client';
import type { Dog } from '../types/models';

export default function DogList() {
  const [dogs, setDogs] = useState<Dog[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchDogs = (query: string) => {
    setLoading(true);
    const params = query ? { search: query } : {};
    api.get<Dog[]>('/dogs/', { params }).then((res) => {
      setDogs(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  };

  useEffect(() => {
    fetchDogs('');
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchDogs(search);
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dogs</h1>

      <form onSubmit={handleSearch} className="mb-4 flex gap-2">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by name..."
          className="border rounded-md px-3 py-2 text-sm flex-1 max-w-sm"
        />
        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700">
          Search
        </button>
      </form>

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : dogs.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 text-lg">No dogs found</p>
          <p className="text-gray-400 text-sm mt-1">Run the scraper to populate dog data</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Sex</th>
                <th className="px-4 py-3">Trainer</th>
                <th className="px-4 py-3">Sire</th>
                <th className="px-4 py-3">Dam</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {dogs.map((dog) => (
                <tr key={dog.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium">{dog.name}</td>
                  <td className="px-4 py-3">{dog.sex || '-'}</td>
                  <td className="px-4 py-3">{dog.trainer_name || '-'}</td>
                  <td className="px-4 py-3 text-gray-500">{dog.sire || '-'}</td>
                  <td className="px-4 py-3 text-gray-500">{dog.dam || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
