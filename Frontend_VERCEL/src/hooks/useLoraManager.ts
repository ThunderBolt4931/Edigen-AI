
import { useState } from 'react';
import { getApiUrl } from "@/config/backend";

export interface LoraItem {
  id: string;
  name: string;
  path: string;
  scale: number;
}

export const useLoraManager = () => {
  const [loras, setLoras] = useState<LoraItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  const addLora = async (name: string, path: string, onError?: (message: string) => void) => {
    if (!path) {
      if (onError) {
        onError("LoRA Path/ID is required");
      }
      return;
    }

    if (!name) {
      if (onError) {
        onError("LoRA Name is required (used as adapter name)");
      }
      return;
    }

    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("lora_path", path);
      formData.append("adapter_name", name);

      const response = await fetch(getApiUrl("/load-lora/"), {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        const newLora: LoraItem = {
          id: Date.now().toString(),
          name,
          path,
          scale: 1.0
        };
        setLoras(prev => [...prev, newLora]);
        
        setShowSuccess(true);
        setTimeout(() => setShowSuccess(false), 2000);
        
        return true;
      } else {
        if (onError) {
          onError(result.detail || result.message || "Failed to load LoRA");
        }
        return false;
      }
    } catch (error) {
      if (onError) {
        onError("Failed to connect to the backend. Make sure your FastAPI server is running on http://localhost:8000");
      }
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const removeLora = (id: string) => {
    setLoras(prev => prev.filter(lora => lora.id !== id));
  };

  const updateLoraScale = (id: string, scale: number) => {
    setLoras(prev => prev.map(lora => 
      lora.id === id ? { ...lora, scale } : lora
    ));
  };

  return {
    loras,
    isLoading,
    showSuccess,
    addLora,
    removeLora,
    updateLoraScale
  };
};
