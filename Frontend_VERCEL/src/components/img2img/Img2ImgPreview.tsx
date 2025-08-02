import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Eye, Maximize2, Image as ImageIcon, Loader2, Play } from "lucide-react";
import { useState } from "react";

interface Img2ImgPreviewProps {
  inputImage: string | null;
  generatedImage: string | null;
  isGenerating: boolean;
  previewMode: 'side-by-side' | 'overlay' | 'stacked';
  onPreviewModeChange: (mode: 'side-by-side' | 'overlay' | 'stacked') => void;
  canGenerate: boolean;
  onGenerate: () => void;
  generationTime: number | null;
}

const Img2ImgPreview = ({
  inputImage,
  generatedImage,
  isGenerating,
  previewMode,
  onPreviewModeChange,
  canGenerate,
  onGenerate,
  generationTime
}: Img2ImgPreviewProps) => {
  const [fullscreen, setFullscreen] = useState(false);

  const renderPreview = () => {
    if (!inputImage) {
      return (
        <div className="w-full h-96 bg-slate-900/50 border-2 border-dashed border-slate-600 rounded-lg flex items-center justify-center">
          <div className="text-center">
            <ImageIcon className="w-16 h-16 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400">Upload an image to see preview</p>
          </div>
        </div>
      );
    }

    if (isGenerating) {
      return (
        <div className="w-full h-96 bg-slate-900/50 rounded-lg flex items-center justify-center relative overflow-hidden">
          <div className="absolute inset-0 shimmer-bg animate-shimmer" />
          <div className="text-center relative z-10">
            <Loader2 className="w-16 h-16 text-green-400 mx-auto mb-4 animate-spin" />
            <p className="text-slate-400">Generating your image...</p>
          </div>
        </div>
      );
    }

    if (previewMode === 'side-by-side') {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-96">
          <div className="space-y-2">
            <p className="text-slate-400 text-sm">Original</p>
            <img
              src={inputImage}
              alt="Original"
              className="w-full h-full object-cover rounded-lg border border-slate-600"
            />
          </div>
          {generatedImage && (
            <div className="space-y-2">
              <p className="text-slate-400 text-sm">Generated</p>
              <img
                src={generatedImage}
                alt="Generated"
                className="w-full h-full object-cover rounded-lg border border-slate-600"
              />
            </div>
          )}
        </div>
      );
    }

    if (previewMode === 'stacked') {
      return (
        <div className="space-y-4">
          <div className="space-y-2">
            <p className="text-slate-400 text-sm">Original</p>
            <img
              src={inputImage}
              alt="Original"
              className="w-full h-48 object-cover rounded-lg border border-slate-600"
            />
          </div>
          {generatedImage && (
            <div className="space-y-2">
              <p className="text-slate-400 text-sm">Generated</p>
              <img
                src={generatedImage}
                alt="Generated"
                className="w-full h-48 object-cover rounded-lg border border-slate-600"
              />
            </div>
          )}
        </div>
      );
    }

    // Overlay mode
    return (
      <div className="relative w-full h-96">
        <img
          src={generatedImage || inputImage}
          alt={generatedImage ? "Generated" : "Original"}
          className="w-full h-full object-cover rounded-lg border border-slate-600"
        />
        {generatedImage && (
          <div className="absolute top-2 left-2">
            <Badge className="bg-green-600/80 text-white">
              Generated
            </Badge>
          </div>
        )}
      </div>
    );
  };

  return (
    <Card className="bg-slate-00/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Eye className="w-5 h-5" />
          Preview & Result Viewer
          <div className="ml-auto flex gap-2">
            <Button
              onClick={() => setFullscreen(!fullscreen)}
              variant="outline"
              size="sm"
              className="bg-slate-700 border-slate-600"
            >
              <Maximize2 className="w-4 h-4" />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Preview Mode Toggle */}
        <div className="flex gap-2">
          <Button
            onClick={() => onPreviewModeChange('side-by-side')}
            variant={previewMode === 'side-by-side' ? 'default' : 'outline'}
            size="sm"
            className={previewMode === 'side-by-side' ? 'bg-[#A3E635]/80' : 'bg-slate-700 border-slate-600'}
          >
            Side-by-Side
          </Button>
          <Button
            onClick={() => onPreviewModeChange('overlay')}
            variant={previewMode === 'overlay' ? 'default' : 'outline'}
            size="sm"
            className={previewMode === 'overlay' ? 'bg-[#A3E635]/80' : 'bg-slate-700 border-slate-600'}
          >
            Overlay
          </Button>
          <Button
            onClick={() => onPreviewModeChange('stacked')}
            variant={previewMode === 'stacked' ? 'default' : 'outline'}
            size="sm"
            className={previewMode === 'stacked' ? 'bg-[#A3E635]/80' : 'bg-slate-700 border-slate-600'}
          >
            Stacked
          </Button>
        </div>

        {/* Preview Area */}
        {renderPreview()}

        {/* Generate Button and Generation Time */}
        <div className="space-y-3 pt-4 border-t border-slate-600">
          <Button
            onClick={onGenerate}
            disabled={!canGenerate || isGenerating}
            className="w-full bg-[#A3E635]/80 hover:bg-green-700 disabled:opacity-50 h-12"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Generate Image
              </>
            )}
          </Button>

          {generationTime && (
            <div className="text-center">
              <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                Generation time: {generationTime.toFixed(1)}s
              </Badge>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default Img2ImgPreview;
