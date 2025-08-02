
import { Layers } from "lucide-react";
import LoraItem from "./LoraItem";
import { LoraItem as LoraItemType } from "@/hooks/useLoraManager";

interface LoraListProps {
  loras: LoraItemType[];
  onRemoveLora: (id: string) => void;
  onUpdateLoraScale: (id: string, scale: number) => void;
}

const LoraList = ({ loras, onRemoveLora, onUpdateLoraScale }: LoraListProps) => {
  if (loras.length === 0) {
    return (
      <div className="text-center py-6 text-slate-400 transition-all duration-300 hover:text-slate-300">
        <Layers className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p>No LoRAs added yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {loras.map((lora, index) => (
        <LoraItem
          key={lora.id}
          lora={lora}
          index={index}
          onRemove={onRemoveLora}
          onUpdateScale={onUpdateLoraScale}
        />
      ))}
    </div>
  );
};

export default LoraList;
