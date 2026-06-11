import axios from 'axios';
import { toast } from 'sonner';

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

/**
 * Extracts a human-readable message from an API error without resorting to
 * `any`. FastAPI returns errors as `{ detail: string }`; anything else
 * (network failure, non-string detail) falls back to the supplied message.
 */
export function errorMessage(e: unknown, fallback: string): string {
  if (axios.isAxiosError(e)) {
    const detail = (e.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === 'string' && detail) return detail;
  }
  return fallback;
}

// Global error surface: every failed API request shows a toast with the
// server's detail message (or a generic fallback), then re-throws so callers
// can still branch on the failure. Callers must NOT toast the same error
// again — catch blocks exist only to stop post-mutation refetches and
// unhandled rejections.
api.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    const message = errorMessage(error, 'Request failed');
    // Keyed by message so polling loops update one toast instead of stacking.
    toast.error(message, { id: `api-error:${message}` });
    return Promise.reject(error);
  },
);

export default api;
