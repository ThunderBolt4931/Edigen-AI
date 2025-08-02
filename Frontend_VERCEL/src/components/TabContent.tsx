
import { TabsContent, Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import PlaygroundContent from "@/components/PlaygroundContent";
import DocumentationPanel from "@/components/DocumentationPanel";
import SystemDashboard from "@/components/SystemDashboard";
import InOutpaintingTab from "@/components/InOutpaintingTab";
import Img2ImgTab from "@/components/Img2ImgTab";
import { ComfyUITab } from "@/components/ComfyUITab";
import { Play, Palette, Image, Workflow } from "lucide-react";

interface TabContentProps {
  prompt: string;
  onPromptChange: (value: string) => void;
  controlImage: File | null;
  onControlImageChange: (file: File | null) => void;
  width: number;
  height: number;
  numInferenceSteps: number;
  guidanceScale: number;
  loraScales: Record<string, number>;
  onWidthChange: (value: number) => void;
  onHeightChange: (value: number) => void;
  onStepsChange: (value: number) => void;
  onGuidanceScaleChange: (value: number) => void;
  onLoraScalesChange: (scales: Record<string, number>) => void;
  isGenerating: boolean;
  generatedImage: string | null;
  showSuccess: boolean;
  onGenerate: () => void;
  onImageLoad: () => void;
  onImageError: (e: React.SyntheticEvent<HTMLImageElement, Event>) => void;
  onError: (message: string) => void;
  activeSubTab: string;
  setActiveSubTab: (tab: string) => void;
}

const TabContent = ({
  prompt,
  onPromptChange,
  controlImage,
  onControlImageChange,
  width,
  height,
  numInferenceSteps,
  guidanceScale,
  loraScales,
  onWidthChange,
  onHeightChange,
  onStepsChange,
  onGuidanceScaleChange,
  onLoraScalesChange,
  isGenerating,
  generatedImage,
  showSuccess,
  onGenerate,
  onImageLoad,
  onImageError,
  onError,
  activeSubTab,
  setActiveSubTab
}: TabContentProps) => {
  return (
    <>
      <TabsContent value="playground" className="space-y-4">
        <Tabs value={activeSubTab} onValueChange={setActiveSubTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-slate-800/50 border-slate-700 mb-6">
            <TabsTrigger 
              value="txtTOimg" 
              className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
            >
              <Play className="w-4 h-4 mr-2" />
              txtTOimg
            </TabsTrigger>
            <TabsTrigger 
              value="inoutpainting" 
              className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
            >
              <Palette className="w-4 h-4 mr-2" />
              Inpainting
            </TabsTrigger>
            <TabsTrigger 
              value="img2img" 
              className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
            >
              <Image className="w-4 h-4 mr-2" />
              img2img
            </TabsTrigger>
            <TabsTrigger 
              value="comfyui" 
              className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
            >
              <Workflow className="w-4 h-4 mr-2" />
              TryOn
            </TabsTrigger>
          </TabsList>

          <TabsContent value="txtTOimg" className="space-y-4">
            <PlaygroundContent
              prompt={prompt}
              onPromptChange={onPromptChange}
              controlImage={controlImage}
              onControlImageChange={onControlImageChange}
              width={width}
              height={height}
              numInferenceSteps={numInferenceSteps}
              guidanceScale={guidanceScale}
              loraScales={loraScales}
              onWidthChange={onWidthChange}
              onHeightChange={onHeightChange}
              onStepsChange={onStepsChange}
              onGuidanceScaleChange={onGuidanceScaleChange}
              onLoraScalesChange={onLoraScalesChange}
              isGenerating={isGenerating}
              generatedImage={generatedImage}
              showSuccess={showSuccess}
              onGenerate={onGenerate}
              onImageLoad={onImageLoad}
              onImageError={onImageError}
              onError={onError}
            />
          </TabsContent>

          <TabsContent value="inoutpainting" className="space-y-4">
            <InOutpaintingTab
              generatedImage={generatedImage}
              onError={onError}
            />
          </TabsContent>

          <TabsContent value="img2img" className="space-y-4">
            <Img2ImgTab
              onError={onError}
            />
          </TabsContent>

          <TabsContent value="comfyui" className="space-y-4">
            <ComfyUITab />
          </TabsContent>
        </Tabs>
      </TabsContent>

      <TabsContent value="documentation" className="space-y-4">
        <DocumentationPanel />
      </TabsContent>

      <TabsContent value="dashboard" className="space-y-4">
        <SystemDashboard />
      </TabsContent>
    </>
  );
};

export default TabContent;
