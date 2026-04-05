export default function Predictions() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Predictions</h1>
      <div className="bg-white rounded-lg shadow p-8 text-center">
        <p className="text-gray-500 text-lg">No predictions yet</p>
        <p className="text-gray-400 text-sm mt-1">
          Train a model and scrape upcoming races to generate predictions
        </p>
      </div>
    </div>
  );
}
