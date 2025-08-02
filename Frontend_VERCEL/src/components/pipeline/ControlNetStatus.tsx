
import { Badge } from "@/components/ui/badge";
import { CheckCircle, Circle } from "lucide-react";

interface ControlNetStatusProps {
  hasError: boolean;
  activeControlnet?: boolean;
}

const ControlNetStatus = ({ hasError, activeControlnet }: ControlNetStatusProps) => {
  const getStatusColor = () => {
    if (hasError) return "destructive";
    if (activeControlnet) return "default";
    return "secondary";
  };

  const getStatusText = () => {
    if (hasError) return "Connection Failed";
    if (activeControlnet) return "ControlNet Active";
    return "No ControlNet";
  };

  const getStatusIcon = () => {
    if (activeControlnet) return <CheckCircle className="w-3 h-3" />;
    return <Circle className="w-3 h-3" />;
  };

  return (
    <div className="flex items-center justify-between p-3 bg-slate-900/30 rounded-lg border border-slate-600">
      <div className="flex items-center gap-3">
        {getStatusIcon()}
        <div>
          <p className="text-white font-medium">ControlNet Status</p>
          <Badge variant={getStatusColor()} className="mt-1">
            {getStatusText()}
          </Badge>
        </div>
      </div>
    </div>
  );
};

export default ControlNetStatus;
