import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import AppHeader from "@/components/AppHeader";
import ErrorManager from "@/components/ErrorManager";
import { useErrorManager } from "@/hooks/useErrorManager";
import { useImageGeneration } from "@/hooks/useImageGeneration";
import PlaygroundContent from "@/components/PlaygroundContent";
import InOutpaintingTab from "@/components/InOutpaintingTab";
import Img2ImgTab from "@/components/Img2ImgTab";
import { ComfyUITab } from "@/components/ComfyUITab";
import { Button } from "@/components/ui/button";
import { Play, Image, Paintbrush, Zap, Monitor } from "lucide-react";

const AppPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [activeTab, setActiveTab] = useState("txtTOimg");
  const { errors, addError, clearError, clearAllErrors } = useErrorManager();
  const {
    prompt,
    isGenerating,
    generatedImage,
    showSuccess,
    width,
    height,
    numInferenceSteps,
    guidanceScale,
    controlImage,
    loraScales,
    setPrompt,
    setWidth,
    setHeight,
    setNumInferenceSteps,
    setGuidanceScale,
    setControlImage,
    setLoraScales,
    handleGenerate,
    handleImageLoad,
    handleImageError
  } = useImageGeneration();

  useEffect(() => {
    const tabParam = searchParams.get('tab');
    
    if (tabParam) {
      const tabMapping: { [key: string]: string } = {
        'txt2img': 'txtTOimg',
        'img2img': 'imgTOimg', 
        'inpainting': 'inpainting',
        'tryon': 'comfyui'
      };
      
      const mappedTab = tabMapping[tabParam] || tabParam;
      setActiveTab(mappedTab);
    }
  }, [searchParams]);

  const onGenerate = (selectedModel?: string) => {
    handleGenerate(addError, selectedModel);
  };

  const handleSystemDashboard = () => {
    navigate('/systemdashboard');
  };

  return (
    <>
      {/* Background Spline 3D iframe */}
      <div className="fixed top-[10px] left-[-120px] w-[120%] h-[120%] z-0">
          <iframe 
              src='https://my.spline.design/retrofuturisticcircuitloop-mMGk86WKmMW9qCpLyPQkAZR0/' 
              frameBorder='0' 
              width='100%' 
              height='100%' 
              className="pointer-events-none"
          />
      </div>
      
      <div className="min-h-screen bg-transparent relative z-10">
        <AppHeader />
        
        <div className="container mx-auto p-4 pt-20">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-4 bg-slate-800/50 border-slate-700">
              <TabsTrigger 
                value="txtTOimg" 
                className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
              >
                <Play className="w-4 h-4 mr-2" />
                Text to Image
              </TabsTrigger>
              <TabsTrigger 
                value="inpainting" 
                className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
              >
                <Paintbrush className="w-4 h-4 mr-2" />
                Inpainting
              </TabsTrigger>
              <TabsTrigger 
                value="imgTOimg" 
                className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
              >
                <Image className="w-4 h-4 mr-2" />
                Image to Image
              </TabsTrigger>
              <TabsTrigger 
                value="comfyui" 
                className="data-[state=active]:bg-[#A3E635]/40 data-[state=active]:text-green-100 text-slate-400"
              >
                <Zap className="w-4 h-4 mr-2" />
                Try-On
              </TabsTrigger>
            </TabsList>

            <TabsContent value="txtTOimg" className="space-y-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-white mb-2">Text to Image</h1>
                  <p className="text-gray-300">Generate stunning images from text prompts with advanced AI models and creative control.</p>
                </div>
                <Button
                  onClick={handleSystemDashboard}
                  className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all bg-green-600/20 text-green-300 hover:bg-green-600/30"
                >
                  <Monitor className="w-4 h-4 mr-2" />
                  System Dashboard
                </Button>
              </div>
              <PlaygroundContent
                prompt={prompt}
                onPromptChange={setPrompt}
                controlImage={controlImage}
                onControlImageChange={setControlImage}
                width={width}
                height={height}
                numInferenceSteps={numInferenceSteps}
                guidanceScale={guidanceScale}
                loraScales={loraScales}
                onWidthChange={setWidth}
                onHeightChange={setHeight}
                onStepsChange={setNumInferenceSteps}
                onGuidanceScaleChange={setGuidanceScale}
                onLoraScalesChange={setLoraScales}
                isGenerating={isGenerating}
                generatedImage={generatedImage}
                showSuccess={showSuccess}
                onGenerate={onGenerate}
                onImageLoad={handleImageLoad}
                onImageError={handleImageError}
                onError={addError}
              />
            </TabsContent>

            <TabsContent value="inpainting" className="space-y-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-white mb-2">Inpainting & Outpainting</h1>
                  <p className="text-gray-300">Intelligently fill, edit, and enhance specific regions of your images with precision.</p>
                </div>
                <Button
                  onClick={handleSystemDashboard}
                  className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all bg-green-600/20 text-green-300 hover:bg-green-600/30"
                >
                  <Monitor className="w-4 h-4 mr-2" />
                  System Dashboard
                </Button>
              </div>
              <InOutpaintingTab 
                generatedImage={generatedImage}
                onError={addError} 
              />
            </TabsContent>

            <TabsContent value="imgTOimg" className="space-y-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold text-white mb-2">Image to Image</h1>
                  <p className="text-gray-300">Transform existing images with AI-powered modifications and style transfers.</p>
                </div>
                <Button
                  onClick={handleSystemDashboard}
                  className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all bg-green-600/20 text-green-300 hover:bg-green-600/30"
                >
                  <Monitor className="w-4 h-4 mr-2" />
                  System Dashboard
                </Button>
              </div>
              <Img2ImgTab onError={addError} />
            </TabsContent>

            <TabsContent value="comfyui" className="space-y-6">
              <div className="mb-6">
                <h1 className="text-3xl font-bold text-white mb-2">Virtual Try-On</h1>
                <p className="text-gray-300">Experience advanced virtual try-on capabilities with ComfyUI workflows.</p>
              </div>
              <ComfyUITab />
            </TabsContent>
          </Tabs>
        </div>

        <ErrorManager
          errors={errors}
          onClearError={clearError}
          onClearAll={clearAllErrors}
        />
      </div>

      {/* CSS for shimmer animation */}
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        .shimmer-bg {
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
          animation: shimmer 2s infinite;
        }
        
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
      `}</style>
    </>
  );
};

export default AppPage;
