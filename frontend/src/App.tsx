import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AppLayout from './components/layout/AppLayout';
import Dashboard from './pages/Dashboard';
import RaceList from './pages/RaceList';
import DogList from './pages/DogList';
import FeatureBuilder from './pages/FeatureBuilder';
import TrainingLab from './pages/TrainingLab';
import Predictions from './pages/Predictions';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/races" element={<RaceList />} />
            <Route path="/dogs" element={<DogList />} />
            <Route path="/features" element={<FeatureBuilder />} />
            <Route path="/training" element={<TrainingLab />} />
            <Route path="/predictions" element={<Predictions />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
