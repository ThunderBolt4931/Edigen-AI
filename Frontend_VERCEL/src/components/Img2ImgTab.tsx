import { useState } from 'react';
import Img2ImgImageInput from '@/components/img2img/Img2ImgImageInput';
import Img2ImgPromptSection from '@/components/img2img/Img2ImgPromptSection';
import Img2ImgModelSelection from '@/components/img2img/Img2ImgModelSelection';
import Img2ImgControls from '@/components/img2img/Img2ImgControls';
import Img2ImgPreview from '@/components/img2img/Img2ImgPreview';
import Img2ImgPostGeneration from '@/components/img2img/Img2ImgPostGeneration';
import { getApiUrl } from '@/config/backend';
import CollapsibleSection from './CollapsibleSection';
import { Settings } from 'lucide-react';

interface Img2ImgTabProps {
  onError: (message: string) => void;
}

const Img2ImgTab = ({ onError }: Img2ImgTabProps) => {
  // Image state
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  
  // Prompt state
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [selectedStyle, setSelectedStyle] = useState('none');
  
  // Model & Adapter state
  const [selectedModel, setSelectedModel] = useState('SD 1.5');
  const [selectedLora, setSelectedLora] = useState('none');
  const [loraScale, setLoraScale] = useState(0.7);
  const [selectedControlNet, setSelectedControlNet] = useState('none');
  const [controlNetEnabled, setControlNetEnabled] = useState(false);
  const [controlNetImage, setControlNetImage] = useState<File | null>(null);
  
  // Generation controls
  const [denoisingStrength, setDenoisingStrength] = useState(0.7);
  const [cfgScale, setCfgScale] = useState(7.5);
  const [inferenceSteps, setInferenceSteps] = useState(20);
  const [seed, setSeed] = useState<number | null>(null);
  const [autoDetectResolution, setAutoDetectResolution] = useState(true);
  
  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationTime, setGenerationTime] = useState<number | null>(null);
  
  // Preview state
  const [previewMode, setPreviewMode] = useState<'side-by-side' | 'overlay' | 'stacked'>('side-by-side');

  const handleGenerate = async () => {
    if (!inputImage || !prompt.trim()) {
      onError('Please provide an input image and prompt');
      return;
    }

    setIsGenerating(true);
    const startTime = Date.now();
    
    try {
      // Create FormData for multipart/form-data request
      const formData = new FormData();
      
      // Add text fields
      formData.append('prompt', prompt);
      formData.append('base_model', selectedModel);
      formData.append('denoising_strength', denoisingStrength.toString());
      formData.append('cfg_scale', cfgScale.toString());
      formData.append('num_inference_steps', inferenceSteps.toString());
      
      // Add optional fields
      if (selectedLora !== 'none') {
        formData.append('lora_adapter', selectedLora);
      }
      if (controlNetEnabled && selectedControlNet !== 'none') {
        formData.append('controlnet_path', selectedControlNet);
      }
      if (seed !== null) {
        formData.append('seed', seed.toString());
      }
      
      // Convert base64 image to blob and add as file
      const response = await fetch(inputImage);
      const blob = await response.blob();
      formData.append('init_image', blob, 'init_image.png');
      
      // Add control image if available
      if (controlNetEnabled && controlNetImage) {
        formData.append('control_image', controlNetImage, 'control_image.png');
      }
      
      // Send request to backend
      const apiResponse = await fetch(getApiUrl('/generate-img2img-image/'), {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true'
        },
        body: formData
      });
      
      if (!apiResponse.ok) {
        throw new Error(`API request failed: ${apiResponse.status}`);
      }
      
      // Handle response - assuming it returns a base64 image
      const responseBlob = await apiResponse.blob();
      const imageUrl = URL.createObjectURL(responseBlob);
      
      setGeneratedImage(imageUrl);
      setGenerationTime((Date.now() - startTime) / 1000);
    } catch (error) {
      console.error('Generation error:', error);
      onError('Failed to generate img2img result');
    } finally {
      setIsGenerating(false);
    }
  };

  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 1000000));
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Controls */}
        <div className="space-y-6">
          {/* Image Input */}
          <Img2ImgImageInput
            inputImage={inputImage}
            onImageChange={setInputImage}
            onError={onError}
          />

          {/* Prompt Section */}
          <Img2ImgPromptSection
            prompt={prompt}
            negativePrompt={negativePrompt}
            selectedStyle={selectedStyle}
            onPromptChange={setPrompt}
            onNegativePromptChange={setNegativePrompt}
            onStyleChange={setSelectedStyle}
          />

          {/* Model & Adapter Selection */}
          <CollapsibleSection
            title="Model & Adapter Selection"
            icon={<Settings className="w-5 h-5" />}
            defaultOpen={false}
          >
            <Img2ImgModelSelection
              selectedModel={selectedModel}
              selectedLora={selectedLora}
              loraScale={loraScale}
              selectedControlNet={selectedControlNet}
              controlNetEnabled={controlNetEnabled}
              controlNetImage={controlNetImage}
              onModelChange={setSelectedModel}
              onLoraChange={setSelectedLora}
              onLoraScaleChange={setLoraScale}
              onControlNetChange={setSelectedControlNet}
              onControlNetEnabledChange={setControlNetEnabled}
              onControlNetImageChange={setControlNetImage}
              onError={onError}
            />
          </CollapsibleSection>

          {/* Generation Controls */}
          <CollapsibleSection
            title="Image-to-Image Controls"
            icon={<Settings className="w-5 h-5" />}
            defaultOpen={false}
          >
            <Img2ImgControls
              denoisingStrength={denoisingStrength}
              cfgScale={cfgScale}
              inferenceSteps={inferenceSteps}
              seed={seed}
              autoDetectResolution={autoDetectResolution}
              onDenoisingStrengthChange={setDenoisingStrength}
              onCfgScaleChange={setCfgScale}
              onInferenceStepsChange={setInferenceSteps}
              onSeedChange={setSeed}
              onAutoDetectResolutionChange={setAutoDetectResolution}
              onRandomizeSeed={randomizeSeed}
              isGenerating={isGenerating}
              canGenerate={!!(inputImage && prompt.trim())}
              onGenerate={handleGenerate}
              generationTime={generationTime}
            />
          </CollapsibleSection>
        </div>

        {/* Right Column - Preview & Results */}
        <div className="space-y-6">
          {/* Preview */}
          <Img2ImgPreview
            inputImage={inputImage}
            generatedImage={generatedImage}
            isGenerating={isGenerating}
            previewMode={previewMode}
            onPreviewModeChange={setPreviewMode}
            canGenerate={!!(inputImage && prompt.trim())}
            onGenerate={handleGenerate}
            generationTime={generationTime}
          />

          {/* Post-Generation Controls */}
          {generatedImage && (
            <Img2ImgPostGeneration
              generatedImage={generatedImage}
              prompt={prompt}
              settings={{
                denoisingStrength,
                cfgScale,
                inferenceSteps,
                seed,
                selectedLora,
                loraScale
              }}
              onError={onError}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Img2ImgTab;
