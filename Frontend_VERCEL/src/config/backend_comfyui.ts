/**
 * ComfyUI Backend Configuration
 * 
 * Update the BACKEND_COMFYUI_URL constant below to change the ComfyUI backend URL.
 * 
 * Example URLs:
 * - Local development: "http://localhost:8188"
 * - ngrok tunnel: "https://your-id.ngrok-free.app"
 * - Production: "https://your-comfyui-domain.com"
 */

export const BACKEND_COMFYUI_URL = "https://your-ngrok-url.ngrok-free.app";

/**
 * ComfyUI backend configuration object with additional settings
 */
export const comfyUIConfig = {
  url: BACKEND_COMFYUI_URL,
  headers: {
    'ngrok-skip-browser-warning': 'true'
  }
} as const;

/**
 * Helper function to get the full ComfyUI API endpoint URL 
 */
export const getComfyUIApiUrl = (endpoint: string): string => {
  return `${BACKEND_COMFYUI_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
};