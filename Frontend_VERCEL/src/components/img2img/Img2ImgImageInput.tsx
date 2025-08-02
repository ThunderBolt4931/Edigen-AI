
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Upload, Image as ImageIcon, Clipboard, Trash2 } from "lucide-react";
import { useRef } from "react";

interface Img2ImgImageInputProps {
  inputImage: string | null;
  onImageChange: (image: string | null) => void;
  onError: (message: string) => void;
}

const Img2ImgImageInput = ({ inputImage, onImageChange, onError }: Img2ImgImageInputProps) => {
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
      onError('Failed to paste from clipboard');
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        onImageChange(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      onError('Please drop a valid image file');
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const clearImage = () => {
    onImageChange(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Card className="bg-slate-00/70 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <ImageIcon className="w-5 h-5" />
          Image Input
          <Badge variant="secondary" className="ml-auto bg-slate-700 text-slate-300">
            {inputImage ? 'Image Loaded' : 'No Image'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {!inputImage ? (
          <div
            className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center transition-colors hover:border-green-500"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <ImageIcon className="w-16 h-16 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400 mb-4">
              Drag & drop an image here, or click to browse
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              <Button
                onClick={() => fileInputRef.current?.click()}
                className="bg-[#A3E635]/80 hover:bg-green-700"
              >
                <Upload className="w-4 h-4 mr-2" />
                Browse Files
              </Button>
              <Button
                onClick={handlePasteFromClipboard}
                variant="outline"
                className="bg-slate-700 border-slate-600"
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
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative">
              <img
                src={inputImage}
                alt="Input image"
                className="w-full h-64 object-cover rounded-lg border border-slate-600"
              />
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                className="bg-slate-700 border-slate-600"
              >
                <Upload className="w-4 h-4 mr-2" />
                Replace Image
              </Button>
              <Button
                onClick={clearImage}
                variant="outline"
                className="bg-slate-700 border-slate-600"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear
              </Button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default Img2ImgImageInput;
