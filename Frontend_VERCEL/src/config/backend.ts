
/**
 * Backend Configuration
 * 
 * Update the BACKEND_URL constant below to change the backend URL for the entire application.
 * This will automatically sync across all components that use the backend.
 * 
 * Example URLs:
 * - Local development: "http://localhost:8000"
 * - ngrok tunnel: "https://0f2d-35-240-133-85.ngrok-free.app"
 * - Production: "https://your-api-domain.com"
 */

export const BACKEND_URL = "https://69be14f97b9c.ngrok-free.app";

/**
 * Backend configuration object with additional settings
 */
export const backendConfig = {
  url: BACKEND_URL,
  headers: {
    'ngrok-skip-browser-warning': 'true'
  }
} as const;

/**
 * Helper function to get the full API endpoint URL 
 */
export const getApiUrl = (endpoint: string): string => {
  return `${BACKEND_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
};
