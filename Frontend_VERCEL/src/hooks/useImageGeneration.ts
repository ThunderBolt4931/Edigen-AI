import { useState } from 'react';
import { useToast } from "@/hooks/use-toast";
import { backendConfig, getApiUrl } from "@/config/backend";

export const useImageGeneration = () => {
  const [prompt, setPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);
  
  // Generation parameters
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [numInferenceSteps, setNumInferenceSteps] = useState(20);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [controlImage, setControlImage] = useState<File | null>(null);
  const [loraScales, setLoraScales] = useState<Record<string, number>>({});
  
  const { toast } = useToast();

  const handleGenerate = async (addError: (message: string) => void, selectedModel?: string) => {
    if (!prompt.trim()) {
      addError("Please enter a prompt");
      return;
    }

    setIsGenerating(true);
    setGeneratedImage(null);

    try {
      console.log("Sending request to backend...");
      
      // Create FormData for multipart request
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('height', height.toString());
      formData.append('width', width.toString());
      formData.append('num_inference_steps', numInferenceSteps.toString());
      formData.append('guidance_scale', guidanceScale.toString());
      
      // Only add control image if one is selected
      if (controlImage) {
        console.log("Adding control image:", controlImage.name, controlImage.size);
        formData.append('control_image', controlImage);
      }
      
      // Add LoRA scales if any
      if (Object.keys(loraScales).length > 0) {
        formData.append('lora_scales', JSON.stringify(loraScales));
      }

      // Add selected model if provided
      if (selectedModel) {
        formData.append('model', selectedModel);
      }

      // Log FormData contents for debugging
      console.log("FormData contents:");
      for (const [key, value] of formData.entries()) {
        console.log(`${key}:`, value);
      }

      // Don't set Content-Type header - let the browser set it automatically for multipart/form-data
      const headers = {
        ...backendConfig.headers
        // Note: DO NOT set 'Content-Type': 'multipart/form-data' here
        // The browser will set it automatically with the correct boundary
      };

      const response = await fetch(getApiUrl('/generate-image/'), {
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
      setGeneratedImage(imageUrl);

      // Show success animation
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 2000);

      toast({
        title: "Success",
        description: "Image generated successfully!",
      });

    } catch (error) {
      console.error('Error generating image:', error);
      addError(`Failed to generate image: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleImageLoad = () => {
    console.log("Image loaded successfully");
  };

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    console.error("Image failed to load:", e);
    toast({
      title: "Error",
      description: "Failed to display the generated image",
      variant: "destructive",
    });
  };

  return {
    // State
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
    
    // Setters
    setPrompt,
    setWidth,
    setHeight,
    setNumInferenceSteps,
    setGuidanceScale,
    setControlImage,
    setLoraScales,
    
    // Handlers
    handleGenerate,
    handleImageLoad,
    handleImageError
  };
};
