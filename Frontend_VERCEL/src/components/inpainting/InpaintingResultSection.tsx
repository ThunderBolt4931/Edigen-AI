
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Image as ImageIcon } from "lucide-react";

interface InpaintingResultSectionProps {
  resultImages: {
    original: string | null;
    masked: string | null;
    generated: string | null;
  };
  isGenerating: boolean;
  mode: 'inpainting' | 'outpainting';
}

const InpaintingResultSection = ({ 
  resultImages, 
  isGenerating, 
  mode 
}: InpaintingResultSectionProps) => {
  const downloadImage = (imageUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card className="bg-slate-00/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <ImageIcon className="w-5 h-5" />
          Results
          <Badge variant="secondary" className={`ml-auto transition-all duration-300 ${
            isGenerating 
              ? "bg-orange-500/20 text-orange-300 animate-pulse" 
              : resultImages.generated 
                ? "bg-green-500/20 text-green-300" 
                : "bg-slate-700 text-slate-300"
          }`}>
            {isGenerating ? "Generating..." : resultImages.generated ? "Complete" : "Idle"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isGenerating ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="aspect-square bg-slate-900/50 border-2 border-dashed border-slate-600 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-pulse relative z-10">
                    <ImageIcon className="w-12 h-12 text-slate-400 mx-auto mb-2" />
                    <p className="text-slate-400 text-sm">Processing...</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : resultImages.generated ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Original */}
              <div className="space-y-2">
                <h4 className="text-slate-300 font-medium text-center">Original</h4>
                <img 
                  src={resultImages.original!} 
                  alt="Original" 
                  className="w-full h-auto rounded-lg border border-slate-600"
                />
              </div>

              {/* Masked */}
              <div className="space-y-2">
                <h4 className="text-slate-300 font-medium text-center">Mask</h4>
                <img 
                  src={resultImages.masked!} 
                  alt="Mask" 
                  className="w-full h-auto rounded-lg border border-slate-600"
                />
              </div>

              {/* Generated */}
              <div className="space-y-2">
                <h4 className="text-slate-300 font-medium text-center">Generated</h4>
                <img 
                  src={resultImages.generated} 
                  alt="Generated" 
                  className="w-full h-auto rounded-lg border border-slate-600"
                />
              </div>
            </div>

            {/* Download Button */}
            <div className="flex justify-center">
              <Button
                onClick={() => downloadImage(resultImages.generated!, `${mode}-result.png`)}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                <Download className="w-4 h-4 mr-2" />
                Download Result
              </Button>
            </div>
          </div>
        ) : (
          <div className="aspect-video bg-slate-900/50 border-2 border-dashed border-slate-600 rounded-lg flex items-center justify-center">
            <div className="text-center">
              <ImageIcon className="w-16 h-16 text-slate-500 mx-auto mb-4" />
              <p className="text-slate-500">Generate to see results</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default InpaintingResultSection;
