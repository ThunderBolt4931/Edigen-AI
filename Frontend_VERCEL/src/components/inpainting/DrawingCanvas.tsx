import { useRef, useEffect, forwardRef, useImperativeHandle } from "react";

interface DrawingCanvasProps {
  inputImage: string;
  mode: 'inpainting' | 'outpainting';
  canvasWidth: number;
  canvasHeight: number;
  extensionPadding: number;
  originalImageDimensions: { width: number; height: number };
  tool: 'brush' | 'eraser';
  brushSize: number;
  zoom: number;
  showMask: boolean;
  onMaskChange: (maskData: string | null) => void;
  onOriginalImageDimensionsChange: (dimensions: { width: number; height: number }) => void;
  onHistoryChange: (maskData: string) => void;
}

export interface DrawingCanvasRef {
  resetMask: () => void;
  restoreFromHistory: (maskData: string) => void;
  generateMaskData: () => string | null;
}

export const DrawingCanvas = forwardRef<DrawingCanvasRef, DrawingCanvasProps>(({
  inputImage,
  mode,
  canvasWidth,
  canvasHeight,
  extensionPadding,
  originalImageDimensions,
  tool,
  brushSize,
  zoom,
  showMask,
  onMaskChange,
  onOriginalImageDimensionsChange,
  onHistoryChange
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);

  const generateMaskData = () => {
    if (!maskCanvasRef.current) return null;
    
    // Create a temporary canvas for the final mask
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return null;
    
    tempCanvas.width = maskCanvasRef.current.width;
    tempCanvas.height = maskCanvasRef.current.height;
    
    // Fill with black background
    tempCtx.fillStyle = '#000000';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Draw the mask (white areas) from the mask canvas
    tempCtx.drawImage(maskCanvasRef.current, 0, 0);
    
    // Return as PNG data URL
    return tempCanvas.toDataURL('image/png');
  };

  const resetMask = () => {
    if (!maskCanvasRef.current) return;
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) return;

    // Reset mask to black
    maskCtx.fillStyle = '#000000';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    const maskData = generateMaskData();
    if (maskData) {
      onHistoryChange(maskData);
    }
  };

  const restoreFromHistory = (maskDataUrl: string) => {
    if (!maskCanvasRef.current) return;
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) return;

    const img = new Image();
    img.onload = () => {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.fillStyle = '#000000';
      maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.drawImage(img, 0, 0);
      onMaskChange(maskDataUrl);
    };
    img.src = maskDataUrl;
  };

  // Expose methods through ref
  useImperativeHandle(ref, () => ({
    resetMask,
    restoreFromHistory,
    generateMaskData
  }));

  useEffect(() => {
    if (canvasRef.current && maskCanvasRef.current && inputImage) {
      const canvas = canvasRef.current;
      const maskCanvas = maskCanvasRef.current;
      const ctx = canvas.getContext('2d');
      const maskCtx = maskCanvas.getContext('2d');
      if (!ctx || !maskCtx) return;

      const img = new Image();
      img.onload = () => {
        onOriginalImageDimensionsChange({ width: img.width, height: img.height });
        
        if (mode === 'outpainting') {
          // For outpainting, create extended canvas
          const extendedWidth = img.width + (extensionPadding * 2);
          const extendedHeight = img.height + (extensionPadding * 2);
          
          canvas.width = extendedWidth;
          canvas.height = extendedHeight;
          maskCanvas.width = extendedWidth;
          maskCanvas.height = extendedHeight;
          
          // Clear canvas and draw image in center
          ctx.fillStyle = '#1e1e2f';
          ctx.fillRect(0, 0, extendedWidth, extendedHeight);
          ctx.drawImage(img, extensionPadding, extensionPadding);
          
          // Initialize mask canvas with black background
          maskCtx.fillStyle = '#000000';
          maskCtx.fillRect(0, 0, extendedWidth, extendedHeight);
          
          // Draw extension area indicators
          ctx.strokeStyle = 'rgba(147, 51, 234, 0.5)';
          ctx.lineWidth = 2;
          ctx.setLineDash([5, 5]);
          ctx.strokeRect(extensionPadding, extensionPadding, img.width, img.height);
        } else {
          // For inpainting, use original image size
          canvas.width = img.width;
          canvas.height = img.height;
          maskCanvas.width = img.width;
          maskCanvas.height = img.height;
          ctx.drawImage(img, 0, 0);
          
          // Initialize mask canvas with black background
          maskCtx.fillStyle = '#000000';
          maskCtx.fillRect(0, 0, img.width, img.height);
        }
        
        const maskData = generateMaskData();
        if (maskData) {
          onHistoryChange(maskData);
        }
      };
      img.src = inputImage;
    }
  }, [inputImage, mode, extensionPadding]);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDrawingRef.current = true;
    draw(e);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current || !maskCanvasRef.current) return;
    
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) return;

    const rect = maskCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;

    maskCtx.globalCompositeOperation = tool === 'brush' ? 'source-over' : 'destination-out';
    maskCtx.beginPath();
    maskCtx.arc(x, y, brushSize / 2, 0, 2 * Math.PI);
    maskCtx.fillStyle = '#FFFFFF'; // White for inpainting areas
    maskCtx.fill();
  };

  const stopDrawing = () => {
    if (isDrawingRef.current) {
      isDrawingRef.current = false;
      const maskData = generateMaskData();
      if (maskData) {
        onHistoryChange(maskData);
      }
    }
  };

  return (
    <div className="flex justify-center overflow-auto bg-slate-900/50 rounded-lg p-4">
      <div className="relative">
        <canvas
          ref={canvasRef}
          style={{ 
            transform: `scale(${zoom})`,
            position: 'absolute',
            top: 0,
            left: 0,
            zIndex: 1
          }}
          className="border border-slate-600 rounded max-w-full"
        />
        <canvas
          ref={maskCanvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          style={{ 
            transform: `scale(${zoom})`,
            cursor: tool === 'brush' ? 'crosshair' : 'pointer',
            opacity: showMask ? 0.5 : 0,
            position: 'relative',
            zIndex: 2,
            mixBlendMode: 'overlay'
          }}
          className="border border-slate-600 rounded max-w-full"
        />
      </div>
    </div>
  );
});

DrawingCanvas.displayName = 'DrawingCanvas';