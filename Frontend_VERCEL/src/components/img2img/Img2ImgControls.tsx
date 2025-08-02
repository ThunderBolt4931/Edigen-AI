import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Settings, Shuffle, Play, Loader2 } from "lucide-react";

interface Img2ImgControlsProps {
  denoisingStrength: number;
  cfgScale: number;
  inferenceSteps: number;
  seed: number | null;
  autoDetectResolution: boolean;
  onDenoisingStrengthChange: (value: number) => void;
  onCfgScaleChange: (value: number) => void;
  onInferenceStepsChange: (value: number) => void;
  onSeedChange: (value: number | null) => void;
  onAutoDetectResolutionChange: (value: boolean) => void;
  onRandomizeSeed: () => void;
  isGenerating: boolean;
  canGenerate: boolean;
  onGenerate: () => void;
  generationTime: number | null;
}

const Img2ImgControls = ({
  denoisingStrength,
  cfgScale,
  inferenceSteps,
  seed,
  autoDetectResolution,
  onDenoisingStrengthChange,
  onCfgScaleChange,
  onInferenceStepsChange,
  onSeedChange,
  onAutoDetectResolutionChange,
  onRandomizeSeed,
  isGenerating,
  canGenerate,
  onGenerate,
  generationTime
}: Img2ImgControlsProps) => {
  const handleSeedChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    onSeedChange(value === '' ? null : parseInt(value) || null);
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Image-to-Image Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Denoising Strength */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-slate-300">Denoising Strength</Label>
            <span className="text-slate-400 text-sm">{denoisingStrength.toFixed(1)}</span>
          </div>
          <Slider
            value={[denoisingStrength]}
            onValueChange={(value) => onDenoisingStrengthChange(value[0])}
            min={0.1}
            max={1.0}
            step={0.1}
            className="cursor-pointer"
          />
          <p className="text-xs text-slate-500">
            Higher values = more change, Lower values = stay closer to original
          </p>
        </div>

        {/* CFG Scale */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-slate-300">CFG Scale</Label>
            <span className="text-slate-400 text-sm">{cfgScale.toFixed(1)}</span>
          </div>
          <Slider
            value={[cfgScale]}
            onValueChange={(value) => onCfgScaleChange(value[0])}
            min={1}
            max={20}
            step={0.5}
            className="cursor-pointer"
          />
        </div>

        {/* Inference Steps */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-slate-300">Inference Steps</Label>
            <span className="text-slate-400 text-sm">{inferenceSteps}</span>
          </div>
          <Slider
            value={[inferenceSteps]}
            onValueChange={(value) => onInferenceStepsChange(value[0])}
            min={1}
            max={100}
            step={1}
            className="cursor-pointer"
          />
        </div>

        {/* Seed */}
        <div className="space-y-2">
          <Label className="text-slate-300">Seed</Label>
          <div className="flex gap-2">
            <Input
              type="number"
              placeholder="Random"
              value={seed || ''}
              onChange={handleSeedChange}
              className="bg-slate-900/50 border-slate-600 text-white"
            />
            <Button
              onClick={onRandomizeSeed}
              variant="outline"
              size="icon"
              className="bg-slate-700 border-slate-600"
            >
              <Shuffle className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Auto-detect Resolution */}
        <div className="flex items-center justify-between">
          <Label className="text-slate-300">Auto-detect Resolution</Label>
          <Switch
            checked={autoDetectResolution}
            onCheckedChange={onAutoDetectResolutionChange}
          />
        </div>
      </CardContent>
    </Card>
  );
};

export default Img2ImgControls;
