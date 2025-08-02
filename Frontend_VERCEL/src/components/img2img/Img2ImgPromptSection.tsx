import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Type } from "lucide-react";

interface Img2ImgPromptSectionProps {
  prompt: string;
  negativePrompt: string;
  selectedStyle: string;
  onPromptChange: (prompt: string) => void;
  onNegativePromptChange: (prompt: string) => void;
  onStyleChange: (style: string) => void;
}

const stylePresets = [
  { value: 'none', label: 'None' },
  { value: 'anime', label: 'Anime' },
  { value: 'cyberpunk', label: 'Cyberpunk' },
  { value: 'artistic', label: 'Artistic' },
  { value: 'photorealistic', label: 'Photorealistic' },
  { value: 'oil-painting', label: 'Oil Painting' },
  { value: 'watercolor', label: 'Watercolor' },
  { value: 'digital-art', label: 'Digital Art' },
  { value: 'fantasy', label: 'Fantasy' },
  { value: 'vintage', label: 'Vintage' }
];

const Img2ImgPromptSection = ({
  prompt,
  negativePrompt,
  selectedStyle,
  onPromptChange,
  onNegativePromptChange,
  onStyleChange
}: Img2ImgPromptSectionProps) => {
  return (
    <Card className="bg-slate-00/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Type className="w-5 h-5" />
          Prompt Section
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="main-prompt" className="text-slate-300">
            Main Prompt
          </Label>
          <Textarea
            id="main-prompt"
            placeholder="Describe what you want to generate..."
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            className="bg-slate-900/50 border-slate-600 text-white placeholder-slate-400 min-h-[100px]"
          />
        </div>
      </CardContent>
    </Card>
  );
};

export default Img2ImgPromptSection;
