
import { useState } from 'react';
import ImageInputSection from '@/components/inpainting/ImageInputSection';
import ModeSelector from '@/components/inpainting/ModeSelector';
import MaskingCanvas from '@/components/inpainting/MaskingCanvas';
import InpaintingPromptInput from '@/components/inpainting/InpaintingPromptInput';
import InpaintingAdvancedSettings from '@/components/inpainting/InpaintingAdvancedSettings';
import InpaintingGenerateButton from '@/components/inpainting/InpaintingGenerateButton';
import InpaintingResultSection from '@/components/inpainting/InpaintingResultSection';
import { useToast } from "@/hooks/use-toast";
import { backendConfig, getApiUrl } from "@/config/backend";

interface InOutpaintingTabProps {
  generatedImage: string | null;
  onError: (message: string) => void;
}

const InOutpaintingTab = ({ generatedImage, onError }: InOutpaintingTabProps) => {
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [mode, setMode] = useState<'inpainting' | 'outpainting'>('inpainting');
  const [maskData, setMaskData] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [steps, setSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState<number | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [resultImages, setResultImages] = useState<{
    original: string | null;
    masked: string | null;
    generated: string | null;
  }>({
    original: null,
    masked: null,
    generated: null
  });

  const { toast } = useToast();

  const dataURLtoFile = (dataurl: string, filename: string): File => {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)?.[1] || '';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  const handleGenerate = async () => {
    if (!inputImage || !maskData || !prompt.trim()) {
      onError('Please provide an image, mask, and prompt');
      return;
    }

    setIsGenerating(true);
    try {
      console.log("Sending inpainting request to backend...");
      
      // Create FormData for multipart request
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('height', '512');
      formData.append('width', '512');
      formData.append('num_inference_steps', steps.toString());
      formData.append('guidance_scale', guidanceScale.toString());
      formData.append('strength', '0.8');
      formData.append('generation_mode', mode);
      
      // Convert input image to File object
      const initImageFile = dataURLtoFile(inputImage, 'init_image.png');
      formData.append('init_image', initImageFile);
      
      // Convert mask to File object
      const maskFile = dataURLtoFile(maskData, 'mask.png');
      formData.append('control_image', maskFile);
      
      // Add seed if provided
      if (seed !== null) {
        formData.append('seed', seed.toString());
      }

      // Log FormData contents for debugging
      console.log("FormData contents:");
      for (const [key, value] of formData.entries()) {
        console.log(`${key}:`, value);
      }

      const headers = {
        ...backendConfig.headers
      };

      const response = await fetch(getApiUrl('/generate-image-inpainting/'), {
        method: 'POST',
        headers: headers,
        body: formData
      });

      console.log("Response status:", response.status);
      console.log("Response headers:", Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Backend error:", errorText);
        throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
      }

      // Convert response to blob
      const imageBlob = await response.blob();
      console.log("Image blob received, size:", imageBlob.size);
      
      // Create object URL from blob
      const imageUrl = URL.createObjectURL(imageBlob);
      console.log("Image URL created:", imageUrl);
      
      setResultImages({
        original: inputImage,
        masked: maskData,
        generated: imageUrl
      });

      toast({
        title: "Success",
        description: "Inpainting completed successfully!",
      });

    } catch (error) {
      console.error('Error generating inpainted image:', error);
      onError(`Failed to generate inpainted image: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Step 1: Image Input */}
      <ImageInputSection
        inputImage={inputImage}
        onImageChange={setInputImage}
        generatedImage={generatedImage}
        onError={onError}
      />

      {/* Step 2: Mode Selector */}
      <ModeSelector
        mode={mode}
        onModeChange={setMode}
      />

      {/* Step 3: Masking Canvas */}
      {inputImage && (
        <MaskingCanvas
          inputImage={inputImage}
          mode={mode}
          onMaskChange={setMaskData}
        />
      )}

      {/* Step 4: Prompt Input */}
      <InpaintingPromptInput
        prompt={prompt}
        onPromptChange={setPrompt}
        mode={mode}
      />

      {/* Step 5: Advanced Settings */}
      <InpaintingAdvancedSettings
        steps={steps}
        guidanceScale={guidanceScale}
        seed={seed}
        onStepsChange={setSteps}
        onGuidanceScaleChange={setGuidanceScale}
        onSeedChange={setSeed}
      />

      {/* Step 6: Generate Button */}
      <InpaintingGenerateButton
        isGenerating={isGenerating}
        canGenerate={!!(inputImage && maskData && prompt.trim())}
        onGenerate={handleGenerate}
        mode={mode}
      />

      {/* Result Section */}
      <InpaintingResultSection
        resultImages={resultImages}
        isGenerating={isGenerating}
        mode={mode}
      />
    </div>
  );
};

export default InOutpaintingTab;