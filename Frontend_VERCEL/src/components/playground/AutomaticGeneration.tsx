import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { ChevronDown, ChevronUp, Zap } from "lucide-react";
import { useState } from "react";
import { getApiUrl } from "@/config/backend";

interface AutomaticGenerationProps {
  prompt: string;
  onError: (message: string) => void;
  setGeneratedImage: (img: string) => void;
  setAutoGenMetadata: (metadata: any) => void;
  selectedModel: string;
}

const AutomaticGeneration = ({ prompt, onError, setGeneratedImage, setAutoGenMetadata, selectedModel }: AutomaticGenerationProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [autoGenSteps, setAutoGenSteps] = useState(4);
  const [autoGenSeed, setAutoGenSeed] = useState<string>('');
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [isAutoGenerating, setIsAutoGenerating] = useState(false);
  const [autoGenError, setAutoGenError] = useState<string>('');

  const handleAutoGenerate = async () => {
    setIsAutoGenerating(true);
    setAutoGenError('');

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('num_inference_steps', autoGenSteps.toString());
      if (autoGenSeed.trim()) {
        formData.append('seed', autoGenSeed);
      }
      formData.append('include_metadata', includeMetadata.toString());
      formData.append('model', selectedModel);

      const response = await fetch(getApiUrl('/generate-image-automatic/'), {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Generation failed: ${errorText}`);
      }

      const result = await response.json();
      console.log('Auto generation result:', result);
      
      const imageUrl = "data:image/png;base64," + result.image_base64;
      setGeneratedImage(imageUrl);
      setAutoGenMetadata(result.metadata);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Auto generation failed';
      setAutoGenError(errorMessage);
    } finally {
      setIsAutoGenerating(false);
    }
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader className="pb-3 mt-[-21px] h-[70px]">
        <Button
          onClick={() => setIsExpanded(!isExpanded)}
          variant="ghost"
          className="w-full flex items-center justify-between p-3 h-auto text-white hover:bg-slate-700/50 rounded-lg transition-all duration-300"
        >
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-purple-400" />
            <span className="text-sm font-medium">⚙️ Automatic Generation</span>
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </Button>
      </CardHeader>

      <div 
        className={`overflow-hidden transition-all duration-300 ease-in-out ${
          isExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <CardContent className="bg-gray-800 mx-4 mb-4 p-4 rounded-lg space-y-4">"
          {/* Number of Steps */}
          <div className="space-y-2">
            <Label className="text-slate-300 text-sm">Number of Steps</Label>
            <Input
              type="number"
              value={autoGenSteps}
              onChange={(e) => setAutoGenSteps(Number(e.target.value))}
              min={1}
              max={20}
              className="bg-slate-900/50 border-slate-600 text-white"
            />
          </div>

          {/* Seed */}
          <div className="space-y-2">
            <Label className="text-slate-300 text-sm">Seed</Label>
            <Input
              type="text"
              value={autoGenSeed}
              onChange={(e) => setAutoGenSeed(e.target.value)}
              placeholder="Leave blank for random"
              className="bg-slate-900/50 border-slate-600 text-white placeholder-slate-400"
            />
          </div>

          {/* Metadata Toggle */}
          <div className="flex items-center space-x-2">
            <Checkbox
              id="include-metadata"
              checked={includeMetadata}
              onCheckedChange={(checked) => setIncludeMetadata(checked as boolean)}
            />
            <Label htmlFor="include-metadata" className="text-slate-300 text-sm">
              Include metadata
            </Label>
          </div>

          {/* Generate Button */}
          <Button
            onClick={handleAutoGenerate}
            disabled={isAutoGenerating}
            className="w-full bg-transparent border-2 border-purple-500 text-purple-400 hover:bg-purple-500/10 transition-colors duration-300"
          >
            <Zap className="w-4 h-4 mr-2" />
            {isAutoGenerating ? 'Generating...' : 'Generate Automatically'}
          </Button>

          {/* Error Display */}
          {autoGenError && (
            <div className="text-red-400 text-sm bg-red-900/20 p-2 rounded border border-red-800">
              {autoGenError}
            </div>
          )}
        </CardContent>
      </div>
    </Card>
  );
};

export default AutomaticGeneration;
