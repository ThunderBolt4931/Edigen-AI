
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Settings, Info, Sliders } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface GenerationControlsProps {
  width: number;
  height: number;
  numInferenceSteps: number;
  guidanceScale: number;
  loraScales: Record<string, number>;
  onWidthChange: (value: number) => void;
  onHeightChange: (value: number) => void;
  onStepsChange: (value: number) => void;
  onGuidanceScaleChange: (value: number) => void;
  onLoraScalesChange: (scales: Record<string, number>) => void;
}

const GenerationControls = ({
  width,
  height,
  numInferenceSteps,
  guidanceScale,
  loraScales,
  onWidthChange,
  onHeightChange,
  onStepsChange,
  onGuidanceScaleChange,
  onLoraScalesChange
}: GenerationControlsProps) => {
  
  const commonSizes = [
    { label: "512×512", width: 512, height: 512 },
    { label: "768×768", width: 768, height: 768 },
    { label: "1024×1024", width: 1024, height: 1024 },
    { label: "512×768", width: 512, height: 768 },
    { label: "768×512", width: 768, height: 512 },
  ];

  const handleSizePreset = (presetWidth: number, presetHeight: number) => {
    onWidthChange(presetWidth);
    onHeightChange(presetHeight);
  };

  const updateLoraScale = (adapterName: string, scale: number) => {
    onLoraScalesChange({
      ...loraScales,
      [adapterName]: scale
    });
  };

  const removeLoraScale = (adapterName: string) => {
    const newScales = { ...loraScales };
    delete newScales[adapterName];
    onLoraScalesChange(newScales);
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Sliders className="w-5 h-5" />
          Generation Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Image Dimensions */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-medium text-slate-300">Image Dimensions</h4>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="w-4 h-4 text-slate-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Output image size in pixels</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          
          {/* Size Presets */}
          <div className="flex flex-wrap gap-2">
            {commonSizes.map((size) => (
              <Button
                key={size.label}
                variant="outline"
                size="sm"
                onClick={() => handleSizePreset(size.width, size.height)}
                className={`text-xs transition-colors ${
                  width === size.width && height === size.height
                    ? 'bg-purple-500/20 border-purple-500 text-purple-300'
                    : 'bg-slate-700 border-slate-600 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {size.label}
              </Button>
            ))}
          </div>

          {/* Custom Dimensions */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Width</label>
              <div className="space-y-2">
                <Slider
                  value={[width]}
                  onValueChange={(value) => onWidthChange(value[0])}
                  min={256}
                  max={1024}
                  step={64}
                  className="w-full"
                />
                <div className="text-xs text-slate-300 text-center">{width}px</div>
              </div>
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Height</label>
              <div className="space-y-2">
                <Slider
                  value={[height]}
                  onValueChange={(value) => onHeightChange(value[0])}
                  min={256}
                  max={1024}
                  step={64}
                  className="w-full"
                />
                <div className="text-xs text-slate-300 text-center">{height}px</div>
              </div>
            </div>
          </div>
        </div>

        <Separator className="bg-slate-600" />

        {/* Generation Parameters */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-medium text-slate-300">Generation Parameters</h4>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="w-4 h-4 text-slate-500" />
                </TooltipTrigger>
                <TooltipContent>
                  <p>Controls for image generation quality and style</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Inference Steps</label>
              <div className="space-y-2">
                <Slider
                  value={[numInferenceSteps]}
                  onValueChange={(value) => onStepsChange(value[0])}
                  min={10}
                  max={50}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-300">
                  <span>{numInferenceSteps} steps</span>
                  <span className="text-slate-500">Higher = better quality, slower</span>
                </div>
              </div>
            </div>

            <div>
              <label className="text-xs text-slate-400 block mb-1">Guidance Scale</label>
              <div className="space-y-2">
                <Slider
                  value={[guidanceScale]}
                  onValueChange={(value) => onGuidanceScaleChange(value[0])}
                  min={1}
                  max={20}
                  step={0.5}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-300">
                  <span>{guidanceScale}</span>
                  <span className="text-slate-500">Higher = follows prompt more closely</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* LoRA Scales */}
        {Object.keys(loraScales).length > 0 && (
          <>
            <Separator className="bg-slate-600" />
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-medium text-slate-300">LoRA Scales</h4>
                <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                  {Object.keys(loraScales).length}
                </Badge>
              </div>
              
              <div className="space-y-3">
                {Object.entries(loraScales).map(([adapterName, scale]) => (
                  <div key={adapterName} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-300 truncate">{adapterName}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeLoraScale(adapterName)}
                        className="h-6 w-6 p-0 text-slate-500 hover:text-red-400"
                      >
                        ×
                      </Button>
                    </div>
                    <div className="space-y-1">
                      <Slider
                        value={[scale]}
                        onValueChange={(value) => updateLoraScale(adapterName, value[0])}
                        min={0}
                        max={2}
                        step={0.1}
                        className="w-full"
                      />
                      <div className="text-xs text-slate-400 text-center">{scale}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default GenerationControls;
