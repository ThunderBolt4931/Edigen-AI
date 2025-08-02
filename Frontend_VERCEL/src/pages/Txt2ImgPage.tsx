import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AppHeader from "@/components/AppHeader";
import ErrorManager from "@/components/ErrorManager";
import { useErrorManager } from "@/hooks/useErrorManager";
import { useImageGeneration } from "@/hooks/useImageGeneration";
import PlaygroundContent from "@/components/PlaygroundContent";
import { Button } from "@/components/ui/button";
import { Monitor } from "lucide-react";

const Txt2ImgPage = () => {
  const navigate = useNavigate();
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

  const handleSystemDashboard = () => {
    navigate('/systemdashboard');
  };

  return (
    <>
      {/* Background Spline 3D iframe */}
      <div className="fixed top-[10px] left-[-86px] w-[120%] h-[120%] z-0">
          <iframe 
              src='https://my.spline.design/retrofuturisticcircuitloop-mMGk86WKmMW9qCpLyPQkAZR0/' 
              frameBorder='0' 
              width='100%' 
              height='100%' 
              className="pointer-events-none"
          />
      </div>
      <AppHeader />
      <div className="min-h-screen bg-transparent relative z-10">
        <div className="container mx-auto p-4 pt-20">
          

          <div className="w-full">
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
          </div>
        </div>

        <ErrorManager
          errors={errors}
          onClearError={clearError}
          onClearAll={clearAllErrors}
        />
      </div>
    </>
  );
};

export default Txt2ImgPage;
