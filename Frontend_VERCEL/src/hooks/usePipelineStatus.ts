
import { useState, useEffect } from 'react';
import { backendConfig, getApiUrl } from "@/config/backend";

interface PipelineState {
  pipeline_type: string;
  active_controlnet: boolean;
  active_adapters: string[];
}

export const usePipelineStatus = (onError?: (message: string) => void) => {
  const [pipelineState, setPipelineState] = useState<PipelineState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchPipelineState = async (showLoadingIndicator = true) => {
    if (showLoadingIndicator) setIsLoading(true);
    try {
      const response = await fetch(getApiUrl('/pipeline-state/'), {
        headers: backendConfig.headers
      });

      if (response.ok) {
        const data = await response.json();
        setPipelineState(data);
        setHasError(false);
        setLastUpdate(new Date());
      } else {
        throw new Error('Failed to fetch pipeline state');
      }
    } catch (err) {
      const errorMessage = 'Failed to connect to pipeline';
      setHasError(true);
      if (onError) {
        onError(errorMessage);
      }
    } finally {
      if (showLoadingIndicator) setIsLoading(false);
    }
  };

  // Auto-refresh every 5 seconds
  useEffect(() => {
    fetchPipelineState();
    const interval = setInterval(() => {
      fetchPipelineState(false); // Silent refresh
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return {
    pipelineState,
    isLoading,
    hasError,
    lastUpdate,
    fetchPipelineState
  };
};
