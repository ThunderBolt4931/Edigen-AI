
import { useState, useEffect, useCallback } from 'react';
import { getApiUrl } from '@/config/backend';

interface CpuRamStats {
  cpu_percent: number;
  ram_total: number;
  ram_used: number;
  ram_percent: number;
}

interface GpuStats {
  gpu_memory_total: number;
  gpu_memory_used: number;
  gpu_utilization: number;
}

interface SystemStats {
  cpu_ram: CpuRamStats;
  gpu: GpuStats;
}

export const useSystemStats = () => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gpuError, setGpuError] = useState(false);
  const [isAutoUpdating, setIsAutoUpdating] = useState(true);
  const [history, setHistory] = useState<SystemStats[]>([]);

  const fetchStats = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      setGpuError(false);

      const response = await fetch(getApiUrl('/stats'), {
        headers: {
          'ngrok-skip-browser-warning': 'true'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch stats: ${response.status}`);
      }

      const data: SystemStats = await response.json();
      
      // Check if GPU data is valid
      if (!data.gpu || data.gpu.gpu_memory_total === undefined) {
        setGpuError(true);
      }

      setStats(data);
      setHistory(prev => [...prev.slice(-29), data]); // Keep last 30 data points
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch system stats');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  useEffect(() => {
    if (!isAutoUpdating) return;

    const interval = setInterval(fetchStats, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, [fetchStats, isAutoUpdating]);

  const toggleAutoUpdate = () => {
    setIsAutoUpdating(prev => !prev);
  };

  const refreshStats = () => {
    fetchStats();
  };

  return {
    stats,
    history,
    isLoading,
    error,
    gpuError,
    isAutoUpdating,
    toggleAutoUpdate,
    refreshStats
  };
};
