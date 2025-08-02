
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";
import { LoraItem as LoraItemType } from "@/hooks/useLoraManager";

interface LoraItemProps {
  lora: LoraItemType;
  index: number;
  onRemove: (id: string) => void;
  onUpdateScale: (id: string, scale: number) => void;
}

const LoraItem = ({ lora, index, onRemove, onUpdateScale }: LoraItemProps) => {
  return (
    <div 
      className="p-3 bg-slate-900/30 rounded-lg border border-slate-600 hover:border-slate-500 transition-all duration-300 hover:shadow-md animate-in slide-in-from-left-2 duration-500" 
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <h4 className="text-white font-medium">{lora.name}</h4>
          <p className="text-sm text-slate-400">{lora.path}</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="bg-slate-700 text-slate-300 transition-all duration-200 hover:bg-slate-600">
            {lora.scale.toFixed(1)}
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onRemove(lora.id)}
            className="text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all duration-200 ease-in-out transform hover:scale-110 active:scale-95"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>
      <div className="space-y-2">
        <Label className="text-slate-300 text-sm">Scale: {lora.scale}</Label>
        <Slider
          value={[lora.scale]}
          onValueChange={(value) => onUpdateScale(lora.id, value[0])}
          max={2}
          min={-2}
          step={0.1}
          className="w-full"
        />
      </div>
    </div>
  );
};

export default LoraItem;
