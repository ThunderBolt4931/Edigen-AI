
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Activity, RefreshCw } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { usePipelineStatus } from "@/hooks/usePipelineStatus";
import PipelineTypeDisplay from "./pipeline/PipelineTypeDisplay";
import ControlNetStatus from "./pipeline/ControlNetStatus";
import ActiveAdapters from "./pipeline/ActiveAdapters";

interface PipelineStatusPanelProps {
  onError?: (message: string) => void;
}

const PipelineStatusPanel = ({ onError }: PipelineStatusPanelProps) => {
  const {
    pipelineState,
    isLoading,
    hasError,
    lastUpdate,
    fetchPipelineState
  } = usePipelineStatus(onError);

  return (
    <Card className="bg-slate-800/50 border-slate-700 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300">
      <CardHeader className="pb-3">
        <CardTitle className="text-white flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Pipeline Status
          </div>
          <div className="flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => fetchPipelineState(true)}
                    disabled={isLoading}
                    className="text-slate-300 hover:text-white hover:bg-slate-700 h-8 w-8 p-0 transition-all duration-200 ease-in-out"
                  >
                    <RefreshCw className={`w-4 h-4 transition-transform duration-200 ${isLoading ? 'animate-spin' : ''}`} />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Refresh pipeline status</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            {lastUpdate && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Badge variant="outline" className="text-xs text-slate-400 border-slate-600">
                      Auto-refresh
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Last updated: {lastUpdate.toLocaleTimeString()}</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Pipeline Type */}
        {pipelineState && (
          <PipelineTypeDisplay pipelineType={pipelineState.pipeline_type} />
        )}

        {/* ControlNet Status */}
        <ControlNetStatus 
          hasError={hasError} 
          activeControlnet={pipelineState?.active_controlnet} 
        />

        {/* Active Adapters */}
        {pipelineState && (
          <ActiveAdapters activeAdapters={pipelineState.active_adapters} />
        )}

        {/* Loading State */}
        {isLoading && !pipelineState && (
          <div className="text-center py-6 text-slate-400">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-slate-400 mx-auto mb-2"></div>
            <p>Loading pipeline status...</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PipelineStatusPanel;
