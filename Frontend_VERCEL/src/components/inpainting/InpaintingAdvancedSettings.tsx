
import CollapsibleSection from "@/components/CollapsibleSection";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Settings } from "lucide-react";

interface InpaintingAdvancedSettingsProps {
  steps: number;
  guidanceScale: number;
  seed: number | null;
  onStepsChange: (steps: number) => void;
  onGuidanceScaleChange: (scale: number) => void;
  onSeedChange: (seed: number | null) => void;
}

const InpaintingAdvancedSettings = ({
  steps,
  guidanceScale,
  seed,
  onStepsChange,
  onGuidanceScaleChange,
  onSeedChange
}: InpaintingAdvancedSettingsProps) => {
  return (
    <CollapsibleSection
      title="Advanced Settings"
      icon={<Settings className="w-5 h-5" />}
      defaultOpen={false}
    >
      <div className="space-y-6">
        {/* Steps */}
        <div className="space-y-2">
          <Label className="text-slate-300">Inference Steps: {steps}</Label>
          <Slider
            value={[steps]}
            onValueChange={(value) => onStepsChange(value[0])}
            min={10}
            max={50}
            step={1}
            className="cursor-pointer"
          />
          <div className="flex justify-between text-xs text-slate-400">
            <span>Faster</span>
            <span>Higher Quality</span>
          </div>
        </div>

        {/* Guidance Scale */}
        <div className="space-y-2">
          <Label className="text-slate-300">Guidance Scale: {guidanceScale}</Label>
          <Slider
            value={[guidanceScale]}
            onValueChange={(value) => onGuidanceScaleChange(value[0])}
            min={1}
            max={20}
            step={0.5}
            className="cursor-pointer"
          />
          <div className="flex justify-between text-xs text-slate-400">
            <span>Creative</span>
            <span>Precise</span>
          </div>
        </div>

        {/* Seed */}
        <div className="space-y-2">
          <Label className="text-slate-300">Seed (optional)</Label>
          <Input
            type="number"
            value={seed || ''}
            onChange={(e) => onSeedChange(e.target.value ? parseInt(e.target.value) : null)}
            placeholder="Random seed"
            className="bg-slate-900/50 border-slate-600 text-white placeholder-slate-400"
          />
          <div className="text-xs text-slate-400">
            Leave empty for random generation
          </div>
        </div>
      </div>
    </CollapsibleSection>
  );
};

export default InpaintingAdvancedSettings;
