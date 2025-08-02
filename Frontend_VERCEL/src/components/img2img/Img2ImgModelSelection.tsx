import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Settings, Upload } from "lucide-react";
import { useRef } from "react";

interface Img2ImgModelSelectionProps {
  selectedModel: string;
  selectedLora: string;
  loraScale: number;
  selectedControlNet: string;
  controlNetEnabled: boolean;
  controlNetImage: File | null;
  onModelChange: (model: string) => void;
  onLoraChange: (lora: string) => void;
  onLoraScaleChange: (scale: number) => void;
  onControlNetChange: (controlNet: string) => void;
  onControlNetEnabledChange: (enabled: boolean) => void;
  onControlNetImageChange: (file: File | null) => void;
  onError: (message: string) => void;
}

const baseModels = [
  { value: 'SD 1.5', label: 'Stable Diffusion 1.5' },
  { value: 'SDXL', label: 'Stable Diffusion XL' },
  { value: 'SD 2.1', label: 'Stable Diffusion 2.1' }
];

const loraOptions = [
  { value: 'none', label: 'None' },
  { value: 'anime-style', label: 'Anime Style' },
  { value: 'portrait-enhance', label: 'Portrait Enhance' },
  { value: 'landscape-boost', label: 'Landscape Boost' }
];

const controlNetOptions = [
  { value: 'none', label: 'None' },
  { value: 'canny', label: 'Canny Edge' },
  { value: 'depth', label: 'Depth Map' },
  { value: 'pose', label: 'Pose Detection' },
  { value: 'normal', label: 'Normal Map' }
];

const Img2ImgModelSelection = ({
  selectedModel,
  selectedLora,
  loraScale,
  selectedControlNet,
  controlNetEnabled,
  controlNetImage,
  onModelChange,
  onLoraChange,
  onLoraScaleChange,
  onControlNetChange,
  onControlNetEnabledChange,
  onControlNetImageChange,
  onError
}: Img2ImgModelSelectionProps) => {
  const controlNetFileRef = useRef<HTMLInputElement>(null);

  const handleControlNetFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        onControlNetImageChange(file);
      } else {
        onError('Please select a valid image file for ControlNet');
      }
    }
  };

  return (
    // To manually adjust the height of this section, change the h-[450px] value below.
    // For example, use h-[500px] to make it taller or h-auto to fit content automatically.
    <Card className="bg-slate-00/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Model & Adapter Selection
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Base Model */}
        <div className="space-y-2">
          <Label className="text-slate-300">Base Model</Label>
          <Select value={selectedModel} onValueChange={onModelChange}>
            <SelectTrigger className="bg-slate-00/50 border-slate-600 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-600">
              {baseModels.map((model) => (
                <SelectItem 
                  key={model.value} 
                  value={model.value}
                  className="text-white hover:bg-slate-700"
                >
                  {model.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* LoRA Section */}
        <div className="space-y-3">
          <Label className="text-slate-300">LoRA Adapter</Label>
          <Select value={selectedLora} onValueChange={onLoraChange}>
            <SelectTrigger className="bg-slate-900/50 border-slate-600 text-white">
              <SelectValue placeholder="Select LoRA..." />
            </SelectTrigger>
            <SelectContent className="bg-slate-800 border-slate-600">
              {loraOptions.map((lora) => (
                <SelectItem 
                  key={lora.value} 
                  value={lora.value}
                  className="text-white hover:bg-slate-700"
                >
                  {lora.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          {selectedLora && selectedLora !== 'none' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-slate-400 text-sm">LoRA Scale</Label>
                <span className="text-slate-400 text-sm">{loraScale.toFixed(1)}</span>
              </div>
              <Slider
                value={[loraScale]}
                onValueChange={(value) => onLoraScaleChange(value[0])}
                min={0.1}
                max={1.0}
                step={0.1}
                className="cursor-pointer"
              />
            </div>
          )}
        </div>

        {/* ControlNet Section */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-slate-300">ControlNet</Label>
            <Switch
              checked={controlNetEnabled}
              onCheckedChange={onControlNetEnabledChange}
            />
          </div>
          
          {controlNetEnabled && (
            <>
              <Select value={selectedControlNet} onValueChange={onControlNetChange}>
                <SelectTrigger className="bg-slate-900/50 border-slate-600 text-white">
                  <SelectValue placeholder="Select ControlNet module..." />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-600">
                  {controlNetOptions.map((controlNet) => (
                    <SelectItem 
                      key={controlNet.value} 
                      value={controlNet.value}
                      className="text-white hover:bg-slate-700"
                    >
                      {controlNet.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {selectedControlNet && selectedControlNet !== 'none' && (
                <div className="space-y-2">
                  <Label className="text-slate-400 text-sm">ControlNet Conditioning Image</Label>
                  <Button
                    onClick={() => controlNetFileRef.current?.click()}
                    variant="outline"
                    className="w-full bg-slate-700 border-slate-600"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    {controlNetImage ? controlNetImage.name : 'Upload ControlNet Image'}
                  </Button>
                  <input
                    ref={controlNetFileRef}
                    type="file"
                    accept="image/*"
                    onChange={handleControlNetFileUpload}
                    className="hidden"
                  />
                </div>
              )}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default Img2ImgModelSelection;
