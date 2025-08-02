import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Settings, Plus, Trash2, RefreshCw, Download, Info, Check, X } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { backendConfig, getApiUrl } from "@/config/backend";

interface ActiveControlNet {
  name: string;
  isLoaded: boolean;
}

const ControlNetManager = () => {
  // Form state
  const [controlnetPath, setControlnetPath] = useState('');
  const [adapterName, setAdapterName] = useState('');
  const [loraWeightsPath, setLoraWeightsPath] = useState('');
  const [torchDtype, setTorchDtype] = useState('float16');
  
  // Loading states
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingList, setIsLoadingList] = useState(false);
  const [isUnloading, setIsUnloading] = useState(false);
  
  // Data state
  const [activeControlNets, setActiveControlNets] = useState<string[]>([]);
  const [selectedForUnload, setSelectedForUnload] = useState('');
  
  // UI state
  const [showSuccess, setShowSuccess] = useState(false);
  
  const { toast } = useToast();

  // Fetch active ControlNets
  const fetchActiveControlNets = async () => {
    setIsLoadingList(true);
    try {
      const response = await fetch(getApiUrl('/active-controlnets/'), {
        headers: backendConfig.headers
      });
      
      if (response.ok) {
        const data = await response.json();
        setActiveControlNets(data.active_controlnets || []);
      } else {
        throw new Error('Failed to fetch active ControlNets');
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to fetch active ControlNets",
        variant: "destructive",
      });
    } finally {
      setIsLoadingList(false);
    }
  };

  // Load ControlNet
  const loadControlNet = async () => {
    if (!controlnetPath.trim()) {
      toast({
        title: "Error",
        description: "ControlNet path is required",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("controlnet_path", controlnetPath);
      if (adapterName.trim()) formData.append("adapter_name", adapterName);
      if (loraWeightsPath.trim()) formData.append("lora_weights_path", loraWeightsPath);
      formData.append("torch_dtype", torchDtype);

      const response = await fetch(getApiUrl('/load-controlnet/'), {
        method: 'POST',
        headers: backendConfig.headers,
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        // Clear form
        setControlnetPath('');
        setAdapterName('');
        setLoraWeightsPath('');
        
        // Show success animation
        setShowSuccess(true);
        setTimeout(() => setShowSuccess(false), 2000);

        toast({
          title: "Success",
          description: result.message || "ControlNet loaded successfully",
        });

        // Refresh the list
        fetchActiveControlNets();
      } else {
        toast({
          title: "Error",
          description: result.detail || "Failed to load ControlNet",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to connect to backend",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Unload ControlNet
  const unloadControlNet = async () => {
    if (!selectedForUnload) return;

    setIsUnloading(true);
    try {
      const formData = new FormData();
      formData.append("model_key", selectedForUnload);

      const response = await fetch(getApiUrl('/unload-controlnet/'), {
        method: 'DELETE',
        headers: backendConfig.headers,
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        toast({
          title: "Success",
          description: result.message || "ControlNet unloaded successfully",
        });
        setSelectedForUnload('');
        fetchActiveControlNets();
      } else {
        toast({
          title: "Error",
          description: result.detail || "Failed to unload ControlNet",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to connect to backend",
        variant: "destructive",
      });
    } finally {
      setIsUnloading(false);
    }
  };

  // Load active ControlNets on component mount
  useEffect(() => {
    fetchActiveControlNets();
  }, []);

  return (
    <Card className="bg-slate-800/50 border-slate-700 hover:shadow-lg hover:shadow-purple-500/10 transition-all duration-300">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Settings className="w-5 h-5" />
          ControlNet Manager
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Info className="w-4 h-4 text-slate-500" />
              </TooltipTrigger>
              <TooltipContent>
                <p>Load and manage ControlNet models for image generation control</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Load ControlNet Section */}
        <div className="space-y-4">
          <h3 className="text-white font-medium flex items-center gap-2">
            <Plus className="w-4 h-4" />
            Load ControlNet
          </h3>
          
          <div className="space-y-3 p-4 bg-slate-900/50 rounded-lg border border-slate-600">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <Label className="text-slate-300 text-sm flex items-center gap-1">
                  ControlNet Path *
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="w-3 h-3 text-slate-500" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Path to the ControlNet model directory or Hugging Face repo</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  value={controlnetPath}
                  onChange={(e) => setControlnetPath(e.target.value)}
                  placeholder="lllyasviel/sd-controlnet-canny"
                  className="bg-slate-800 border-slate-600 text-white"
                  disabled={isLoading}
                />
              </div>
              
              <div>
                <Label className="text-slate-300 text-sm flex items-center gap-1">
                  Adapter Name
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="w-3 h-3 text-slate-500" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Optional unique name for this ControlNet instance</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  value={adapterName}
                  onChange={(e) => setAdapterName(e.target.value)}
                  placeholder="my_canny_controlnet"
                  className="bg-slate-800 border-slate-600 text-white"
                  disabled={isLoading}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <Label className="text-slate-300 text-sm flex items-center gap-1">
                  LoRA Weights Path
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="w-3 h-3 text-slate-500" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Optional: Path to LoRA weights for ControlLoRA functionality</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  value={loraWeightsPath}
                  onChange={(e) => setLoraWeightsPath(e.target.value)}
                  placeholder="path/to/lora_weights.safetensors"
                  className="bg-slate-800 border-slate-600 text-white"
                  disabled={isLoading}
                />
              </div>
              
              <div>
                <Label className="text-slate-300 text-sm flex items-center gap-1">
                  Torch Data Type
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="w-3 h-3 text-slate-500" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>float16 for GPU efficiency, float32 for higher precision</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Select value={torchDtype} onValueChange={setTorchDtype} disabled={isLoading}>
                  <SelectTrigger className="bg-slate-800 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="float16">float16 (Recommended)</SelectItem>
                    <SelectItem value="float32">float32</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="relative">
              <Button
                onClick={loadControlNet}
                disabled={!controlnetPath.trim() || isLoading}
                className="w-full bg-blue-600 hover:bg-blue-700 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-blue-500/50"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Loading ControlNet...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4 mr-2" />
                    Load ControlNet
                  </>
                )}
              </Button>

              {showSuccess && (
                <div className="absolute inset-0 flex items-center justify-center bg-green-500/20 rounded-md animate-in fade-in-0 duration-300">
                  <Check className="w-5 h-5 text-green-400 animate-in scale-in-0 duration-500" />
                </div>
              )}
            </div>
          </div>
        </div>

        <Separator className="bg-slate-600" />

        {/* Active ControlNets Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-white font-medium flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Active ControlNets
              <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                {activeControlNets.length}
              </Badge>
            </h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={fetchActiveControlNets}
              disabled={isLoadingList}
              className="text-slate-300 hover:text-white hover:bg-slate-700"
            >
              <RefreshCw className={`w-4 h-4 ${isLoadingList ? 'animate-spin' : ''}`} />
            </Button>
          </div>

          <div className="space-y-2">
            {isLoadingList ? (
              <div className="text-center py-4 text-slate-400">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-slate-400 mx-auto mb-2"></div>
                Loading active ControlNets...
              </div>
            ) : activeControlNets.length === 0 ? (
              <div className="text-center py-6 text-slate-400">
                <Settings className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No ControlNets loaded</p>
              </div>
            ) : (
              activeControlNets.map((controlnet) => (
                <div
                  key={controlnet}
                  className="flex items-center justify-between p-3 bg-slate-900/30 rounded-lg border border-slate-600 hover:border-slate-500 transition-all duration-200"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-white font-medium">{controlnet}</span>
                  </div>
                  <Badge variant="secondary" className="bg-green-500/20 text-green-300">
                    Active
                  </Badge>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Unload ControlNet Section */}
        {activeControlNets.length > 0 && (
          <>
            <Separator className="bg-slate-600" />
            <div className="space-y-4">
              <h3 className="text-white font-medium flex items-center gap-2">
                <Trash2 className="w-4 h-4" />
                Unload ControlNet
              </h3>
              
              <div className="flex gap-3">
                <Select value={selectedForUnload} onValueChange={setSelectedForUnload}>
                  <SelectTrigger className="bg-slate-800 border-slate-600 text-white flex-1">
                    <SelectValue placeholder="Select ControlNet to unload" />
                  </SelectTrigger>
                  <SelectContent>
                    {activeControlNets.map((controlnet) => (
                      <SelectItem key={controlnet} value={controlnet}>
                        {controlnet}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button
                      variant="destructive"
                      disabled={!selectedForUnload || isUnloading}
                      className="hover:scale-105 transition-transform duration-200"
                    >
                      {isUnloading ? (
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent className="z-[9999] bg-slate-800 border-slate-700">
                    <AlertDialogHeader>
                      <AlertDialogTitle className="text-white">Confirm Unload</AlertDialogTitle>
                      <AlertDialogDescription className="text-slate-300">
                        Are you sure you want to unload "{selectedForUnload}"? This action cannot be undone.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel className="bg-slate-700 text-white border-slate-600">
                        Cancel
                      </AlertDialogCancel>
                      <AlertDialogAction
                        onClick={unloadControlNet}
                        className="bg-red-600 hover:bg-red-700"
                      >
                        Unload
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default ControlNetManager;
