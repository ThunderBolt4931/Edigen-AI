import { useState } from 'react';
import { Tabs } from "@/components/ui/tabs";
import TabNavigation from "@/components/TabNavigation";
import TabContent from "@/components/TabContent";
import AppHeader from "@/components/AppHeader";
import ErrorManager from "@/components/ErrorManager";
import { useErrorManager } from "@/hooks/useErrorManager";
import { useImageGeneration } from "@/hooks/useImageGeneration";
import InOutpaintingTab from "@/components/InOutpaintingTab";

const Index = () => {
  const [activeTab, setActiveTab] = useState("playground");
  const [activeSubTab, setActiveSubTab] = useState("txtTOimg");
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

  const onGenerate = (selectedModel?: string) => {
    handleGenerate(addError, selectedModel);
  };

  return (
    <>
      {/* Background Spline 3D iframe */}
      <div className="fixed inset-0 z-0">
        <iframe 
          src='https://my.spline.design/dotwaves-FR8HGbfFmjLusADTS6cmnWBP/' 
          frameBorder='0' 
          width='100%' 
          height='100%'
          className="pointer-events-none"
        />
      </div>
      
      <div className="min-h-screen bg-transparent relative z-10">
        <div className="container mx-auto p-4">
          <AppHeader />

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabNavigation activeTab={activeTab} />

            <TabContent
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
              activeSubTab={activeSubTab}
              setActiveSubTab={setActiveSubTab}
            />
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

export default Index;
