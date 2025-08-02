
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Layers, Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useLoraManager } from "@/hooks/useLoraManager";
import LoraForm from "./lora/LoraForm";
import LoraList from "./lora/LoraList";

const LoraSection = ({ onError }: { onError?: (message: string) => void }) => {
  const {
    loras,
    isLoading,
    showSuccess,
    addLora,
    removeLora,
    updateLoraScale
  } = useLoraManager();

  const handleAddLora = async (name: string, path: string) => {
    return await addLora(name, path, onError);
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Layers className="w-5 h-5" />
          LoRAs
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Info className="w-4 h-4 text-slate-500" />
              </TooltipTrigger>
              <TooltipContent>
                <p>Low-Rank Adaptation models for fine-tuning</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <LoraForm
          onAddLora={handleAddLora}
          isLoading={isLoading}
          showSuccess={showSuccess}
        />

        <LoraList
          loras={loras}
          onRemoveLora={removeLora}
          onUpdateLoraScale={updateLoraScale}
        />
      </CardContent>
    </Card>
  );
};

export default LoraSection;
