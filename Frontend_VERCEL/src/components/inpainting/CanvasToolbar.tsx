
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Brush, Eraser, RotateCcw, Undo2, Redo2, Eye, EyeOff, ZoomIn, ZoomOut } from "lucide-react";

interface CanvasToolbarProps {
  tool: 'brush' | 'eraser';
  brushSize: number;
  showMask: boolean;
  zoom: number;
  canUndo: boolean;
  canRedo: boolean;
  onToolChange: (tool: 'brush' | 'eraser') => void;
  onBrushSizeChange: (size: number) => void;
  onShowMaskToggle: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onReset: () => void;
}

export const CanvasToolbar = ({
  tool,
  brushSize,
  showMask,
  zoom,
  canUndo,
  canRedo,
  onToolChange,
  onBrushSizeChange,
  onShowMaskToggle,
  onZoomIn,
  onZoomOut,
  onUndo,
  onRedo,
  onReset
}: CanvasToolbarProps) => {
  return (
    <div className="flex flex-wrap items-center gap-4">
      <div className="flex gap-2">
        <Button
          onClick={() => onToolChange('brush')}
          variant={tool === 'brush' ? 'default' : 'outline'}
          size="sm"
          className={tool === 'brush' ? 'bg-purple-600' : 'bg-slate-700 border-slate-600'}
        >
          <Brush className="w-4 h-4" />
        </Button>
        <Button
          onClick={() => onToolChange('eraser')}
          variant={tool === 'eraser' ? 'default' : 'outline'}
          size="sm"
          className={tool === 'eraser' ? 'bg-purple-600' : 'bg-slate-700 border-slate-600'}
        >
          <Eraser className="w-4 h-4" />
        </Button>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-slate-300 text-sm">Brush Size:</span>
        <div className="w-24">
          <Slider
            value={[brushSize]}
            onValueChange={(value) => onBrushSizeChange(value[0])}
            min={5}
            max={50}
            step={1}
            className="cursor-pointer"
          />
        </div>
        <span className="text-slate-400 text-sm">{brushSize}px</span>
      </div>

      <div className="flex gap-2">
        <Button
          onClick={onUndo}
          disabled={!canUndo}
          variant="outline"
          size="sm"
          className="bg-slate-700 border-slate-600 disabled:opacity-50"
        >
          <Undo2 className="w-4 h-4" />
        </Button>
        <Button
          onClick={onRedo}
          disabled={!canRedo}
          variant="outline"
          size="sm"
          className="bg-slate-700 border-slate-600 disabled:opacity-50"
        >
          <Redo2 className="w-4 h-4" />
        </Button>
        <Button
          onClick={onReset}
          variant="outline"
          size="sm"
          className="bg-slate-700 border-slate-600"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
      </div>

      <div className="flex gap-2">
        <Button
          onClick={onShowMaskToggle}
          variant="outline"
          size="sm"
          className="bg-slate-700 border-slate-600"
        >
          {showMask ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
        </Button>
        <Button
          onClick={onZoomIn}
          variant="outline"
          size="sm"
          className="bg-slate-700 border-slate-600"
        >
          <ZoomIn className="w-4 h-4" />
        </Button>
        <Button
          onClick={onZoomOut}
          variant="outline"
          size="sm"
          className="bg-slate-700 border-slate-600"
        >
          <ZoomOut className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
};