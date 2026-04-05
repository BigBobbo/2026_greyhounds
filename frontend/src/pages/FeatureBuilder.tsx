import { useEffect, useState } from 'react';
import api from '../api/client';
import type { FeatureDefinition } from '../types/models';

export default function FeatureBuilder() {
  const [features, setFeatures] = useState<FeatureDefinition[]>([]);
  const [activeTab, setActiveTab] = useState<'visual' | 'code'>('visual');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get<FeatureDefinition[]>('/features/').then((res) => {
      setFeatures(res.data);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Feature Builder</h1>

      <div className="flex gap-4">
        {/* Feature list sidebar */}
        <div className="w-72 shrink-0">
          <div className="bg-white rounded-lg shadow p-4">
            <h2 className="font-semibold text-sm mb-3">Defined Features</h2>
            {loading ? (
              <p className="text-gray-500 text-sm">Loading...</p>
            ) : features.length === 0 ? (
              <p className="text-gray-400 text-sm">No features defined yet</p>
            ) : (
              <ul className="space-y-2">
                {features.map((f) => (
                  <li key={f.id} className="flex items-center justify-between text-sm border rounded-md p-2">
                    <div>
                      <p className="font-medium">{f.display_name || f.name}</p>
                      <p className="text-xs text-gray-400">{f.feature_type}</p>
                    </div>
                    <span className={`w-2 h-2 rounded-full ${f.enabled ? 'bg-green-500' : 'bg-gray-300'}`} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Builder area */}
        <div className="flex-1 bg-white rounded-lg shadow p-5">
          <div className="flex border-b mb-4">
            <button
              onClick={() => setActiveTab('visual')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'visual' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Visual Builder
            </button>
            <button
              onClick={() => setActiveTab('code')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'code' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Code Editor
            </button>
          </div>

          {activeTab === 'visual' ? (
            <div className="space-y-4">
              <p className="text-sm text-gray-500">
                Build features using form controls. Select a metric, aggregation, and window to create a new feature.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Metric</label>
                  <select className="border rounded-md px-3 py-2 text-sm w-full">
                    <option>finish_time</option>
                    <option>finish_position</option>
                    <option>weight_kg</option>
                    <option>sp_decimal</option>
                    <option>beaten_distance</option>
                    <option>sectional_time</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Aggregation</label>
                  <select className="border rounded-md px-3 py-2 text-sm w-full">
                    <option>mean</option>
                    <option>median</option>
                    <option>min</option>
                    <option>max</option>
                    <option>stdev</option>
                    <option>count</option>
                    <option>win_rate</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Window (last N races)</label>
                  <input type="number" defaultValue={5} min={1} max={50} className="border rounded-md px-3 py-2 text-sm w-full" />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Filter</label>
                  <select className="border rounded-md px-3 py-2 text-sm w-full">
                    <option>No filter</option>
                    <option>Same track</option>
                    <option>Same distance</option>
                    <option>Same grade</option>
                  </select>
                </div>
              </div>
              <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700">
                Create Feature
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-sm text-gray-500">
                Write a Python function to compute custom features. The function receives the dog's race history and race context.
              </p>
              <div className="border rounded-md bg-gray-900 text-gray-100 p-4 font-mono text-sm min-h-[300px]">
                <pre>{`def compute(dog_history: pd.DataFrame, race_context: dict) -> float:
    """
    dog_history columns: race_date, track_name, distance_m, grade,
        trap, finish_position, finish_time, sectional_time, weight_kg, sp_decimal
    race_context keys: track_id, distance_m, grade, trap, race_date
    """
    recent = dog_history.tail(5)
    if len(recent) == 0:
        return float('nan')
    return recent['finish_time'].mean()`}</pre>
              </div>
              <p className="text-xs text-gray-400">
                Monaco editor will be integrated in the next phase. For now, this shows the expected function signature.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
