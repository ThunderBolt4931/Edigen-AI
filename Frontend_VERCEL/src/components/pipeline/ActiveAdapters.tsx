
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface ActiveAdaptersProps {
  activeAdapters: string[];
}

const ActiveAdapters = ({ activeAdapters }: ActiveAdaptersProps) => {
  if (activeAdapters.length === 0) return null;

  return (
    <>
      <Separator className="bg-slate-600" />
      <div className="space-y-3">
        <p className="text-white font-medium flex items-center gap-2">
          Active Adapters
          <Badge variant="secondary" className="bg-slate-700 text-slate-300">
            {activeAdapters.length}
          </Badge>
        </p>
        <div className="flex flex-wrap gap-2">
          {activeAdapters.map((adapter, index) => (
            <TooltipProvider key={index}>
              <Tooltip>
                <TooltipTrigger>
                  <Badge 
                    variant="outline" 
                    className="bg-purple-500/20 text-purple-300 border-purple-500/50 hover:bg-purple-500/30 cursor-default transition-colors duration-200"
                  >
                    {adapter}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Active adapter: {adapter}</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ))}
        </div>
      </div>
    </>
  );
};

export default ActiveAdapters;
