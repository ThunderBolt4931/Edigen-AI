
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Settings } from "lucide-react";

interface CanvasSizeControlsProps {
  canvasWidth: number;
  canvasHeight: number;
  extensionPadding: number;
  onCanvasWidthChange: (width: number) => void;
  onCanvasHeightChange: (height: number) => void;
  onExtensionPaddingChange: (padding: number) => void;
  onApplySize: () => void;
}

export const CanvasSizeControls = ({
  canvasWidth,
  canvasHeight,
  extensionPadding,
  onCanvasWidthChange,
  onCanvasHeightChange,
  onExtensionPaddingChange,
  onApplySize
}: CanvasSizeControlsProps) => {
  return (
    <Card className="bg-slate-900/50 border-slate-600">
      <CardHeader className="pb-3">
        <CardTitle className="text-white text-sm flex items-center gap-2">
          <Settings className="w-4 h-4" />
          Canvas Size Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label className="text-slate-300 text-sm">Canvas Width</Label>
            <Input
              type="number"
              value={canvasWidth}
              onChange={(e) => onCanvasWidthChange(parseInt(e.target.value) || 512)}
              className="bg-slate-800 border-slate-600 text-white"
            />
          </div>
          <div className="space-y-2">
            <Label className="text-slate-300 text-sm">Canvas Height</Label>
            <Input
              type="number"
              value={canvasHeight}
              onChange={(e) => onCanvasHeightChange(parseInt(e.target.value) || 512)}
              className="bg-slate-800 border-slate-600 text-white"
            />
          </div>
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-slate-300 text-sm">Extension Padding</Label>
            <span className="text-slate-400 text-sm">{extensionPadding}px</span>
          </div>
          <Slider
            value={[extensionPadding]}
            onValueChange={(value) => onExtensionPaddingChange(value[0])}
            min={64}
            max={256}
            step={32}
            className="cursor-pointer"
          />
        </div>
        
        <Button
          onClick={onApplySize}
          className="w-full bg-purple-600 hover:bg-purple-700"
        >
          Apply Canvas Size
        </Button>
      </CardContent>
    </Card>
  );
};