import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Copy, Send, RotateCcw, Save } from "lucide-react";

interface Img2ImgPostGenerationProps {
  generatedImage: string;
  prompt: string;
  settings: {
    denoisingStrength: number;
    cfgScale: number;
    inferenceSteps: number;
    seed: number | null;
    selectedLora: string;
    loraScale: number;
  };
  onError: (message: string) => void;
}

const Img2ImgPostGeneration = ({
  generatedImage,
  prompt,
  settings,
  onError
}: Img2ImgPostGenerationProps) => {
  const handleDownload = () => {
    try {
      const link = document.createElement('a');
      link.href = generatedImage;
      link.download = `img2img-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      onError('Failed to download image');
    }
  };

  const handleCopyToClipboard = async () => {
    try {
      const response = await fetch(generatedImage);
      const blob = await response.blob();
      await navigator.clipboard.write([
        new ClipboardItem({ [blob.type]: blob })
      ]);
    } catch (error) {
      onError('Failed to copy image to clipboard');
    }
  };

  const handleSendToInpainting = () => {
    // This would be implemented to pass the image to the inpainting tab
    onError('Send to inpainting feature coming soon');
  };

  const handleRegenerateWithSameSettings = () => {
    // This would trigger regeneration with current settings
    onError('Regenerate feature coming soon');
  };

  const handleSaveToHistory = () => {
    // This would save the current generation to history
    const historyItem = {
      image: generatedImage,
      prompt,
      settings,
      timestamp: Date.now()
    };
    
    try {
      const history = JSON.parse(localStorage.getItem('img2img-history') || '[]');
      history.unshift(historyItem);
      // Keep only last 20 items
      history.splice(20);
      localStorage.setItem('img2img-history', JSON.stringify(history));
    } catch (error) {
      onError('Failed to save to history');
    }
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Download className="w-5 h-5" />
          Post-Generation Controls
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Action Buttons */}
        <div className="grid grid-cols-2 gap-2">
          <Button
            onClick={handleDownload}
            className="bg-purple-600 hover:bg-purple-700"
          >
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
          
          <Button
            onClick={handleCopyToClipboard}
            variant="outline"
            className="bg-slate-700 border-slate-600"
          >
            <Copy className="w-4 h-4 mr-2" />
            Copy
          </Button>
          
          <Button
            onClick={handleSendToInpainting}
            variant="outline"
            className="bg-slate-700 border-slate-600"
          >
            <Send className="w-4 h-4 mr-2" />
            To Inpainting
          </Button>
          
          <Button
            onClick={handleRegenerateWithSameSettings}
            variant="outline"
            className="bg-slate-700 border-slate-600"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Regenerate
          </Button>
        </div>

        <Button
          onClick={handleSaveToHistory}
          variant="outline"
          className="w-full bg-slate-700 border-slate-600"
        >
          <Save className="w-4 h-4 mr-2" />
          Save to History
        </Button>

        {/* Generation Info */}
        <div className="space-y-2 pt-4 border-t border-slate-600">
          <p className="text-slate-400 text-sm font-medium">Generation Settings:</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Denoising:</span>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300 text-xs">
                {settings.denoisingStrength.toFixed(1)}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">CFG Scale:</span>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300 text-xs">
                {settings.cfgScale.toFixed(1)}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Steps:</span>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300 text-xs">
                {settings.inferenceSteps}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Seed:</span>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300 text-xs">
                {settings.seed || 'Random'}
              </Badge>
            </div>
          </div>
          
          {settings.selectedLora && (
            <div className="flex justify-between">
              <span className="text-slate-500 text-xs">LoRA:</span>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300 text-xs">
                {settings.selectedLora} ({settings.loraScale.toFixed(1)})
              </Badge>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default Img2ImgPostGeneration;
