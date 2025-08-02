
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { FileText } from "lucide-react";

interface InpaintingPromptInputProps {
  prompt: string;
  onPromptChange: (prompt: string) => void;
  mode: 'inpainting' | 'outpainting';
}

const InpaintingPromptInput = ({ 
  prompt, 
  onPromptChange, 
  mode 
}: InpaintingPromptInputProps) => {
  const placeholder = mode === 'inpainting' 
    ? "Describe what you want to appear in the masked areas..."
    : "Describe what you want to extend beyond the image borders...";

  return (
    <Card className="bg-slate-00/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <FileText className="w-5 h-5" />
          Prompt Input
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Textarea
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          placeholder={placeholder}
          className="bg-slate-900/50 border-slate-600 text-white placeholder-slate-400 min-h-[100px] resize-none"
          rows={4}
        />
        <div className="mt-2 text-right text-slate-400 text-sm">
          {prompt.length} characters
        </div>
      </CardContent>
    </Card>
  );
};

export default InpaintingPromptInput;
