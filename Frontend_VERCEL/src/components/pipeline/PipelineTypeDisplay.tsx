
import { Zap, Activity } from "lucide-react";

interface PipelineTypeDisplayProps {
  pipelineType: string;
}

const PipelineTypeDisplay = ({ pipelineType }: PipelineTypeDisplayProps) => {
  const getPipelineIcon = (pipelineType: string) => {
    if (pipelineType.includes('Stable')) return <Zap className="w-4 h-4" />;
    return <Activity className="w-4 h-4" />;
  };

  return (
    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-lg border border-slate-600">
      <div className="flex items-center gap-3">
        {getPipelineIcon(pipelineType)}
        <div>
          <p className="text-white font-medium">Pipeline Type</p>
          <p className="text-slate-300 text-sm">{pipelineType}</p>
        </div>
      </div>
    </div>
  );
};

export default PipelineTypeDisplay;
