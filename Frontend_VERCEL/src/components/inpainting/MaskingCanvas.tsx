import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brush } from "lucide-react";
import { useState, useRef } from "react";
import { CanvasSizeControls } from "./CanvasSizeControls";
import { CanvasToolbar } from "./CanvasToolbar";
import { DrawingCanvas, DrawingCanvasRef } from "./DrawingCanvas";

interface MaskingCanvasProps {
  inputImage: string;
  mode: 'inpainting' | 'outpainting';
  onMaskChange: (maskData: string | null) => void;
}

const MaskingCanvas = ({ inputImage, mode, onMaskChange }: MaskingCanvasProps) => {
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush');
  const [brushSize, setBrushSize] = useState(20);
  const [showMask, setShowMask] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  
  // Canvas size controls for outpainting
  const [canvasWidth, setCanvasWidth] = useState(512);
  const [canvasHeight, setCanvasHeight] = useState(512);
  const [extensionPadding, setExtensionPadding] = useState(128);
  const [originalImageDimensions, setOriginalImageDimensions] = useState({ width: 0, height: 0 });

  const drawingCanvasRef = useRef<DrawingCanvasRef>(null);

  const saveToHistory = (maskData: string) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(maskData);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
    onMaskChange(maskData);
  };

  const undo = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      if (drawingCanvasRef.current) {
        drawingCanvasRef.current.restoreFromHistory(history[newIndex]);
      }
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      if (drawingCanvasRef.current) {
        drawingCanvasRef.current.restoreFromHistory(history[newIndex]);
      }
    }
  };

  const resetMask = () => {
    if (drawingCanvasRef.current) {
      drawingCanvasRef.current.resetMask();
    }
  };

  const updateCanvasSize = () => {
    // This will trigger a re-render of the DrawingCanvas component
    // which will handle the canvas resize logic
  };

  const handleZoomIn = () => setZoom(Math.min(zoom + 0.2, 3));
  const handleZoomOut = () => setZoom(Math.max(zoom - 0.2, 0.5));

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Brush className="w-5 h-5" />
          Masking Canvas
          <Badge variant="secondary" className="ml-auto bg-slate-700 text-slate-300">
            {mode === 'inpainting' ? 'Paint areas to replace (white)' : 'Paint areas to extend (white)'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Canvas Size Controls for Outpainting */}
        {mode === 'outpainting' && (
          <CanvasSizeControls
            canvasWidth={canvasWidth}
            canvasHeight={canvasHeight}
            extensionPadding={extensionPadding}
            onCanvasWidthChange={setCanvasWidth}
            onCanvasHeightChange={setCanvasHeight}
            onExtensionPaddingChange={setExtensionPadding}
            onApplySize={updateCanvasSize}
          />
        )}

        {/* Tools */}
        <CanvasToolbar
          tool={tool}
          brushSize={brushSize}
          showMask={showMask}
          zoom={zoom}
          canUndo={historyIndex > 0}
          canRedo={historyIndex < history.length - 1}
          onToolChange={setTool}
          onBrushSizeChange={setBrushSize}
          onShowMaskToggle={() => setShowMask(!showMask)}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onUndo={undo}
          onRedo={redo}
          onReset={resetMask}
        />

        {/* Canvas Container */}
        <DrawingCanvas
          ref={drawingCanvasRef}
          inputImage={inputImage}
          mode={mode}
          canvasWidth={canvasWidth}
          canvasHeight={canvasHeight}
          extensionPadding={extensionPadding}
          originalImageDimensions={originalImageDimensions}
          tool={tool}
          brushSize={brushSize}
          zoom={zoom}
          showMask={showMask}
          onMaskChange={onMaskChange}
          onOriginalImageDimensionsChange={setOriginalImageDimensions}
          onHistoryChange={saveToHistory}
        />

        <div className="text-center text-slate-400 text-sm">
          Click and drag to {tool === 'brush' ? 'paint mask (white = inpaint)' : 'erase mask'} • 
          Zoom: {Math.round(zoom * 100)}% • 
          Canvas: {canvasWidth}x{canvasHeight}
          {mode === 'outpainting' && (
            <span> • Original: {originalImageDimensions.width}x{originalImageDimensions.height}</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default MaskingCanvas;
