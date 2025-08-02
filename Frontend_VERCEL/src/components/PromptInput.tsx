
import { useState } from 'react';
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings, Info, Upload, X } from "lucide-react";

interface PromptInputProps {
  prompt: string;
  onPromptChange: (prompt: string) => void;
  controlImage: File | null;
  onControlImageChange: (file: File | null) => void;  
}

const PromptInput = ({ prompt, onPromptChange, controlImage, onControlImageChange }: PromptInputProps) => {
  const [isPromptFocused, setIsPromptFocused] = useState(false);

  const handleControlImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onControlImageChange(file);
    }
  };

  const removeControlImage = () => {
    onControlImageChange(null);
  };

  return (
    <Card className="bg-slate-00/50 border-slate-700 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Input
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-sm font-medium text-slate-300 mb-2 block flex items-center gap-2">
            Prompt
            <Info className="w-4 h-4 text-slate-500" />
          </label>
          <div className="relative">
            <Textarea
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              onFocus={() => setIsPromptFocused(true)}
              onBlur={() => setIsPromptFocused(false)}
              placeholder="Describe what you want to generate..."
              className={`min-h-[120px] bg-slate-900/50 border-slate-600 text-white placeholder:text-slate-400 resize-none transition-all duration-300 ${
                isPromptFocused 
                  ? 'border-purple-500 shadow-lg shadow-purple-500/25 scale-[1.02]' 
                  : 'hover:border-slate-500'
              }`}
            />
            {isPromptFocused && (
              <div className="absolute inset-0 rounded-md border-2 border-purple-500 animate-pulse pointer-events-none" />
            )}
          </div>
        </div>

        {/* ControlNet Image Upload */}
        <div>
          <label className="text-sm font-medium text-slate-300 mb-2 block flex items-center gap-2">
            ControlNet Image (Optional)
            <Info className="w-4 h-4 text-slate-500" />
          </label>
          {controlImage ? (
            <div className="relative">
              <img 
                src={URL.createObjectURL(controlImage)} 
                alt="Control image" 
                className="w-full h-32 object-cover rounded-lg border border-slate-600"
              />
              <Button
                size="sm"
                variant="destructive"
                onClick={removeControlImage}
                className="absolute top-2 right-2"
              >
                <X className="w-4 h-4" />
              </Button>
              <p className="text-xs text-slate-400 mt-1">{controlImage.name}</p>
            </div>
          ) : (
            <div className="border-2 border-dashed border-slate-600 rounded-lg p-4 text-center hover:border-slate-500 transition-colors">
              <input
                type="file"
                accept="image/*"
                onChange={handleControlImageUpload}
                className="hidden"
                id="control-image-upload"
              />
              <label 
                htmlFor="control-image-upload" 
                className="cursor-pointer flex flex-col items-center gap-2"
              >
                <Upload className="w-6 h-6 text-slate-500" />
                <span className="text-sm text-slate-400">Click to upload control image</span>
              </label>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default PromptInput;
