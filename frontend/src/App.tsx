import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import AppLayout from './components/layout/AppLayout';
import Dashboard from './pages/Dashboard';
import RaceList from './pages/RaceList';
import RaceDetail from './pages/RaceDetail';
import DogList from './pages/DogList';
import DogProfile from './pages/DogProfile';
import FeatureBuilder from './pages/FeatureBuilder';
import TrainingLab from './pages/TrainingLab';
import ExperimentDetail from './pages/ExperimentDetail';
import Predictions from './pages/Predictions';
import ScrapingStatus from './pages/ScrapingStatus';
import BankrollDashboard from './pages/BankrollDashboard';
import Schedule from './pages/Schedule';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Toaster richColors position="top-right" />
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/races" element={<RaceList />} />
            <Route path="/races/:id" element={<RaceDetail />} />
            <Route path="/dogs" element={<DogList />} />
            <Route path="/dogs/:id" element={<DogProfile />} />
            <Route path="/features" element={<FeatureBuilder />} />
            <Route path="/training" element={<TrainingLab />} />
            <Route path="/training/:id" element={<ExperimentDetail />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/bankroll" element={<BankrollDashboard />} />
            <Route path="/schedule" element={<Schedule />} />
            <Route path="/scraping" element={<ScrapingStatus />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
