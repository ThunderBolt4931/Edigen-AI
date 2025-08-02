
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Upload, Image as ImageIcon, Clipboard } from "lucide-react";
import { useRef } from "react";

interface ImageInputSectionProps {
  inputImage: string | null;
  onImageChange: (image: string | null) => void;
  generatedImage: string | null;
  onError: (message: string) => void;
}

const ImageInputSection = ({ 
  inputImage, 
  onImageChange, 
  generatedImage, 
  onError 
}: ImageInputSectionProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          onImageChange(e.target?.result as string);
        };
        reader.readAsDataURL(file);
      } else {
        onError('Please select a valid image file');
      }
    }
  };

  const handleUseLastGenerated = () => {
    if (generatedImage) {
      onImageChange(generatedImage);
    } else {
      onError('No generated image available');
    }
  };

  const handlePasteFromClipboard = async () => {
    try {
      const clipboardItems = await navigator.clipboard.read();
      for (const clipboardItem of clipboardItems) {
        for (const type of clipboardItem.types) {
          if (type.startsWith('image/')) {
            const blob = await clipboardItem.getType(type);
            const reader = new FileReader();
            reader.onload = (e) => {
              onImageChange(e.target?.result as string);
            };
            reader.readAsDataURL(blob);
            return;
          }
        }
      }
      onError('No image found in clipboard');
    } catch (error) {
      onError('Failed to access clipboard');
    }
  };

  return (
    <Card className="bg-slate-00/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <ImageIcon className="w-5 h-5" />
          Image Input
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Button
            onClick={() => fileInputRef.current?.click()}
            variant="outline"
            className="bg-slate-00 border-slate-600 text-slate-300 hover:bg-slate-600"
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Image
          </Button>
          
          <Button
            onClick={handleUseLastGenerated}
            disabled={!generatedImage}
            variant="outline"
            className="bg-slate-00 border-slate-600 text-slate-300 hover:bg-slate-600 disabled:opacity-50"
          >
            <ImageIcon className="w-4 h-4 mr-2" />
            Use Last Generated
          </Button>
          
          <Button
            onClick={handlePasteFromClipboard}
            variant="outline"
            className="bg-slate-00 border-slate-600 text-slate-300 hover:bg-slate-600"
          >
            <Clipboard className="w-4 h-4 mr-2" />
            Paste from Clipboard
          </Button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />

        {inputImage && (
          <div className="mt-4">
            <img
              src={inputImage}
              alt="Input"
              className="max-w-full h-auto max-h-64 rounded-lg border border-slate-600"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ImageInputSection;
