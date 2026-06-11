import axios from 'axios';

const baseURL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api';

if (import.meta.env.PROD && !import.meta.env.VITE_API_URL) {
  // Without VITE_API_URL every request hits the static host's /api and 404s.
  console.error(
    'VITE_API_URL is not set — API requests will target the static host and fail. ' +
      'Set VITE_API_URL in the Vercel project environment.'
  );
}

const headers: Record<string, string> = { 'Content-Type': 'application/json' };
if (import.meta.env.VITE_API_KEY) {
  headers['X-API-Key'] = import.meta.env.VITE_API_KEY;
}

const api = axios.create({
  baseURL,
  headers,
});

export default api;
