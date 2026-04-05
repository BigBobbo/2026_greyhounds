import { useEffect, useState } from 'react';
import api from '../api/client';
import type { Experiment } from '../types/models';

export default function TrainingLab() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get<Experiment[]>('/training/experiments').then((res) => {
      setExperiments(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Training Lab</h1>
        <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700">
          New Experiment
        </button>
      </div>

      {loading ? (
        <p className="text-gray-500">Loading...</p>
      ) : experiments.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-8 text-center">
          <p className="text-gray-500 text-lg">No experiments yet</p>
          <p className="text-gray-400 text-sm mt-1">
            Create features first, then train a model to start predicting
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-50 text-gray-600 uppercase text-xs">
              <tr>
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Algorithm</th>
                <th className="px-4 py-3">Target</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Key Metric</th>
                <th className="px-4 py-3">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium">{exp.name}</td>
                  <td className="px-4 py-3">{exp.algorithm}</td>
                  <td className="px-4 py-3">{exp.target}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs ${
                      exp.status === 'completed' ? 'bg-green-100 text-green-700' :
                      exp.status === 'running' ? 'bg-yellow-100 text-yellow-700' :
                      exp.status === 'failed' ? 'bg-red-100 text-red-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {exp.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono text-xs">
                    {exp.metrics ? Object.entries(exp.metrics).slice(0, 1).map(([k, v]) => `${k}: ${Number(v).toFixed(4)}`).join('') : '-'}
                  </td>
                  <td className="px-4 py-3 text-gray-500">{exp.created_at?.split('T')[0] || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
