
import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Plus, Check } from "lucide-react";

interface LoraFormProps {
  onAddLora: (name: string, path: string) => Promise<boolean>;
  isLoading: boolean;
  showSuccess: boolean;
}

const LoraForm = ({ onAddLora, isLoading, showSuccess }: LoraFormProps) => {
  const [newLoraName, setNewLoraName] = useState('');
  const [newLoraPath, setNewLoraPath] = useState('');

  const handleSubmit = async () => {
    const success = await onAddLora(newLoraName, newLoraPath);
    if (success) {
      setNewLoraName('');
      setNewLoraPath('');
    }
  };

  return (
    <div className="space-y-3 p-4 bg-slate-900/50 rounded-lg border border-slate-600 hover:border-slate-500 transition-all duration-300 hover:shadow-md">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label className="text-slate-300 text-sm">Name (Adapter Name) *</Label>
          <Input
            value={newLoraName}
            onChange={(e) => setNewLoraName(e.target.value)}
            placeholder="LoRA adapter name"
            className="bg-slate-800 border-slate-600 text-white transition-all duration-300 hover:border-slate-500 focus:border-purple-500 focus:shadow-lg focus:shadow-purple-500/25"
            disabled={isLoading}
          />
        </div>
        <div>
          <Label className="text-slate-300 text-sm">Path/ID *</Label>
          <Input
            value={newLoraPath}
            onChange={(e) => setNewLoraPath(e.target.value)}
            placeholder="model/path or ID"
            className="bg-slate-800 border-slate-600 text-white transition-all duration-300 hover:border-slate-500 focus:border-purple-500 focus:shadow-lg focus:shadow-purple-500/25"
            disabled={isLoading}
          />
        </div>
      </div>
      <div className="relative">
        <Button 
          onClick={handleSubmit}
          className="w-full bg-purple-600 hover:bg-purple-700 transition-all duration-200 ease-in-out transform hover:scale-[1.02] active:scale-[0.98]"
          disabled={!newLoraPath || !newLoraName || isLoading}
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Loading LoRA...
            </>
          ) : (
            <>
              <Plus className="w-4 h-4 mr-2" />
              Add LoRA
            </>
          )}
        </Button>
        
        {showSuccess && (
          <div className="absolute inset-0 flex items-center justify-center bg-green-500/20 rounded-md animate-in fade-in-0 duration-300">
            <Check className="w-5 h-5 text-green-400 animate-in scale-in-0 duration-500" />
          </div>
        )}
      </div>
    </div>
  );
};

export default LoraForm;
