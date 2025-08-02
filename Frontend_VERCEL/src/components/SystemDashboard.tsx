
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Cpu, HardDrive, Monitor, RefreshCw, Play, Pause } from "lucide-react";
import { useSystemStats } from "@/hooks/useSystemStats";
import StatsChart from "@/components/StatsChart";

const SystemDashboard = () => {
  const {
    stats,
    history,
    isLoading,
    error,
    gpuError,
    isAutoUpdating,
    toggleAutoUpdate,
    refreshStats
  } = useSystemStats();

  const formatBytes = (bytes: number): string => {
    return (bytes / (1024 * 1024 * 1024)).toFixed(2);
  };

  const formatMB = (mb: number): string => {
    return mb.toFixed(0);
  };

  const formatMBtoGB = (mb: number): string => {
    if (typeof mb !== 'number' || isNaN(mb)) return '0.00';
    return (mb / 1024).toFixed(2);
  };

  if (error) {
    return (
      <div className="">
        <div className="container mx-auto">
          <Card className="bg-red-900/20 border-red-700">
            <CardContent className="p-6">
              <div className="text-center">
                <h2 className="text-xl font-semibold text-red-300 mb-2">Error Loading System Stats</h2>
                <p className="text-red-200 mb-4">{error}</p>
                <Button onClick={refreshStats} variant="outline" className="text-red-300 border-red-700">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Retry
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="space-y-6">

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* CPU Card */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-blue-400" />
                  CPU Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-2xl font-bold text-white">
                      {stats.cpu_ram.cpu_percent.toFixed(1)}%
                    </span>
                    <Badge 
                      variant="secondary" 
                      className={
                        stats.cpu_ram.cpu_percent > 80 
                          ? "bg-red-500/20 text-red-300" 
                          : stats.cpu_ram.cpu_percent > 60 
                            ? "bg-yellow-500/20 text-yellow-300" 
                            : "bg-green-500/20 text-green-300"
                      }
                    >
                      {stats.cpu_ram.cpu_percent > 80 ? "High" : stats.cpu_ram.cpu_percent > 60 ? "Medium" : "Normal"}
                    </Badge>
                  </div>
                  <Progress value={stats.cpu_ram.cpu_percent} className="h-2" />
                </div>
              </CardContent>
            </Card>

            {/* RAM Card */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2">
                  <HardDrive className="w-5 h-5 text-green-400" />
                  RAM Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="text-center"> {/* Centering the main display text */}
                    <span className="text-2xl font-bold text-white">
                      {formatBytes(stats.cpu_ram.ram_used)} GB / {formatBytes(stats.cpu_ram.ram_total)} GB
                    </span>
                  </div>
                  <Progress value={stats.cpu_ram.ram_percent} className="h-2" />
                  <div className="text-sm text-slate-400 text-center">
                    ({stats.cpu_ram.ram_percent.toFixed(1)}% used)
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* GPU Card */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2">
                  <Monitor className="w-5 h-5 text-purple-400" />
                  GPU Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                {gpuError ? (
                  <div className="text-center py-4">
                    <p className="text-slate-400 mb-2">No GPU detected</p>
                    <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                      N/A
                    </Badge>
                  </div>
                ) : (() => {
                  const gpuMemoryUsedGB = formatMBtoGB(stats.gpu.gpu_memory_used);
                  const gpuMemoryTotalGB = formatMBtoGB(stats.gpu.gpu_memory_total);
                  const gpuMemoryPercentage = stats.gpu.gpu_memory_total > 0 ? (stats.gpu.gpu_memory_used / stats.gpu.gpu_memory_total) * 100 : 0;

                  return (
                    <div className="space-y-3">
                      <div className="text-center"> {/* Centering the main display text */}
                        <span className="text-2xl font-bold text-white">
                          {gpuMemoryUsedGB} GB / {gpuMemoryTotalGB} GB
                        </span>
                      </div>
                      <Progress value={gpuMemoryPercentage} className="h-2" />
                      <div className="text-sm text-slate-400 text-center">
                        ({gpuMemoryPercentage.toFixed(1)}% used)
                      </div>
                      <div className="text-sm text-slate-400">
                        Utilization: {stats.gpu.gpu_utilization.toFixed(1)}%
                      </div>
                    </div>
                  );
                })()}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Charts Section */}
        {history.length > 0 && (
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white">Performance History</CardTitle>
            </CardHeader>
            <CardContent>
              <StatsChart history={history} gpuError={gpuError} />
            </CardContent>
          </Card>
        )}

        {/* Control Buttons */}
        <div className="flex justify-center gap-4">
          <Button 
            onClick={refreshStats} 
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button 
            onClick={toggleAutoUpdate}
            variant="outline"
            className={`${
              isAutoUpdating 
                ? 'bg-green-600/20 border-green-600 text-green-300 hover:bg-green-600/30' 
                : 'bg-gray-600/20 border-gray-600 text-gray-300 hover:bg-gray-600/30'
            }`}
          >
            {isAutoUpdating ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Pause Auto-Update
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Resume Auto-Update
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default SystemDashboard;
