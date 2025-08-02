
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

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

interface StatsChartProps {
  history: SystemStats[];
  gpuError: boolean;
}

const StatsChart = ({ history, gpuError }: StatsChartProps) => {
  const chartData = history.map((stats, index) => ({
    time: index + 1,
    cpu: stats.cpu_ram.cpu_percent,
    ram: stats.cpu_ram.ram_percent,
    gpu: gpuError ? 0 : stats.gpu.gpu_utilization,
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 shadow-lg">
          <p className="text-slate-300 mb-2">{`Sample ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.dataKey.toUpperCase()}: ${entry.value.toFixed(1)}%`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="time" 
            stroke="#9CA3AF"
            tick={{ fill: '#9CA3AF' }}
          />
          <YAxis 
            stroke="#9CA3AF"
            tick={{ fill: '#9CA3AF' }}
            domain={[0, 100]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ color: '#9CA3AF' }}
          />
          <Line 
            type="monotone" 
            dataKey="cpu" 
            stroke="#3B82F6" 
            strokeWidth={2}
            dot={{ fill: '#3B82F6', strokeWidth: 2, r: 3 }}
            name="CPU"
          />
          <Line 
            type="monotone" 
            dataKey="ram" 
            stroke="#10B981" 
            strokeWidth={2}
            dot={{ fill: '#10B981', strokeWidth: 2, r: 3 }}
            name="RAM"
          />
          {!gpuError && (
            <Line 
              type="monotone" 
              dataKey="gpu" 
              stroke="#8B5CF6" 
              strokeWidth={2}
              dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 3 }}
              name="GPU"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default StatsChart;
