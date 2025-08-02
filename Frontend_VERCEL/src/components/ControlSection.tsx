
import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Plus, X, Settings, Grid, Image, Info, ChevronDown, ChevronUp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface ControlLoRA {
  id: string;
  name: string;
  strength: number;
}

interface ControlNet {
  id: string;
  name: string;
  type: string;
  strength: number;
  startStep: number;
  endStep: number;
  imageUrl?: string;
  controlLoRA?: ControlLoRA;
}

interface ControlUnion {
  id: string;
  name: string;
  controlNets: ControlNet[];
  mode: 'more_prompt' | 'more_control' | 'balanced';
}

const ControlSection = () => {
  const [controlNets, setControlNets] = useState<ControlNet[]>([]);
  const [controlUnions, setControlUnions] = useState<ControlUnion[]>([]);
  const [expandedNets, setExpandedNets] = useState<Set<string>>(new Set());

  const controlTypes = [
    'canny', 'depth', 'pose', 'normal', 'semantic', 'scribble', 'lineart', 'mlsd', 'tile'
  ];

  const controlLoRATypes = [
    'lineart', 'sketch', 'anime', 'realistic', 'artistic', 'detailed'
  ];

  const addControlNet = (controlType: string) => {
    const newControlNet: ControlNet = {
      id: Date.now().toString(),
      name: `${controlType}_${Date.now()}`,
      type: controlType,
      strength: 1.0,
      startStep: 0,
      endStep: 1000
    };

    setControlNets([...controlNets, newControlNet]);
    setExpandedNets(prev => new Set([...prev, newControlNet.id]));
  };

  const removeControlNet = (id: string) => {
    setControlNets(controlNets.filter(item => item.id !== id));
    setExpandedNets(prev => {
      const newSet = new Set(prev);
      newSet.delete(id);
      return newSet;
    });
  };

  const updateControlNet = (id: string, field: string, value: any) => {
    setControlNets(controlNets.map(item => 
      item.id === id ? { ...item, [field]: value } : item
    ));
  };

  const addControlLoRA = (controlNetId: string, loraType: string) => {
    const newLoRA: ControlLoRA = {
      id: Date.now().toString(),
      name: `${loraType}_lora`,
      strength: 1.0
    };

    updateControlNet(controlNetId, 'controlLoRA', newLoRA);
  };

  const removeControlLoRA = (controlNetId: string) => {
    updateControlNet(controlNetId, 'controlLoRA', undefined);
  };

  const updateControlLoRA = (controlNetId: string, field: string, value: any) => {
    const controlNet = controlNets.find(net => net.id === controlNetId);
    if (controlNet?.controlLoRA) {
      const updatedLoRA = { ...controlNet.controlLoRA, [field]: value };
      updateControlNet(controlNetId, 'controlLoRA', updatedLoRA);
    }
  };

  const toggleExpanded = (id: string) => {
    setExpandedNets(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  const addControlUnion = () => {
    const newUnion: ControlUnion = {
      id: Date.now().toString(),
      name: `Union_${Date.now()}`,
      controlNets: [],
      mode: 'balanced'
    };
    setControlUnions([...controlUnions, newUnion]);
  };

  const ControlNetComponent = ({ controlNet }: { controlNet: ControlNet }) => {
    const isExpanded = expandedNets.has(controlNet.id);

    return (
      <div className="border border-slate-600 rounded-lg bg-slate-900/30 animate-in slide-in-from-left-2 duration-500">
        <Collapsible open={isExpanded} onOpenChange={() => toggleExpanded(controlNet.id)}>
          <CollapsibleTrigger className="w-full">
            <div className="p-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors duration-200">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full animate-pulse" />
                <div>
                  <h4 className="text-white font-medium text-left">{controlNet.name}</h4>
                  <Badge variant="secondary" className="mt-1 bg-blue-500/20 text-blue-300">
                    {controlNet.type}
                  </Badge>
                </div>
                {controlNet.controlLoRA && (
                  <Badge variant="secondary" className="bg-purple-500/20 text-purple-300">
                    + LoRA
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeControlNet(controlNet.id);
                  }}
                  className="text-red-400 hover:text-red-300 hover:bg-red-500/10 transition-all duration-200"
                >
                  <X className="w-4 h-4" />
                </Button>
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4 text-slate-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-slate-400" />
                )}
              </div>
            </div>
          </CollapsibleTrigger>

          <CollapsibleContent className="animate-accordion-down">
            <div className="p-4 pt-0 space-y-4">
              {/* ControlNet Settings */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-slate-300 text-sm">Strength: {controlNet.strength}</Label>
                  <Slider
                    value={[controlNet.strength]}
                    onValueChange={(value) => updateControlNet(controlNet.id, 'strength', value[0])}
                    max={2}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                </div>
                <div>
                  <Label className="text-slate-300 text-sm">Control Image</Label>
                  <Input
                    type="file"
                    accept="image/*"
                    className="bg-slate-800 border-slate-600 text-white text-sm"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label className="text-slate-300 text-sm">Start Step: {controlNet.startStep}</Label>
                  <Slider
                    value={[controlNet.startStep]}
                    onValueChange={(value) => updateControlNet(controlNet.id, 'startStep', value[0])}
                    max={1000}
                    min={0}
                    step={10}
                    className="w-full"
                  />
                </div>
                <div>
                  <Label className="text-slate-300 text-sm">End Step: {controlNet.endStep}</Label>
                  <Slider
                    value={[controlNet.endStep]}
                    onValueChange={(value) => updateControlNet(controlNet.id, 'endStep', value[0])}
                    max={1000}
                    min={0}
                    step={10}
                    className="w-full"
                  />
                </div>
              </div>

              {/* ControlLoRA Section */}
              <div className="border-t border-slate-600 pt-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <h5 className="text-slate-300 font-medium">ControlLoRA Enhancement</h5>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger>
                          <Info className="w-4 h-4 text-slate-500" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Optional LoRA to enhance this ControlNet</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  
                  {!controlNet.controlLoRA ? (
                    <Select onValueChange={(value) => addControlLoRA(controlNet.id, value)}>
                      <SelectTrigger className="w-40 bg-slate-700 border-slate-600 text-sm">
                        <SelectValue placeholder="Add LoRA" />
                      </SelectTrigger>
                      <SelectContent>
                        {controlLoRATypes.map(type => (
                          <SelectItem key={type} value={type}>
                            {type.charAt(0).toUpperCase() + type.slice(1)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeControlLoRA(controlNet.id)}
                      className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    >
                      <X className="w-4 h-4 mr-1" />
                      Remove LoRA
                    </Button>
                  )}
                </div>

                {controlNet.controlLoRA && (
                  <div className="bg-slate-800/50 rounded-lg p-3 animate-in slide-in-from-top-2 duration-300">
                    <div className="flex items-center justify-between mb-3">
                      <Badge variant="secondary" className="bg-purple-500/20 text-purple-300">
                        {controlNet.controlLoRA.name}
                      </Badge>
                    </div>
                    <div>
                      <Label className="text-slate-300 text-sm">
                        LoRA Strength: {controlNet.controlLoRA.strength}
                      </Label>
                      <Slider
                        value={[controlNet.controlLoRA.strength]}
                        onValueChange={(value) => updateControlLoRA(controlNet.id, 'strength', value[0])}
                        max={2}
                        min={0}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Control Systems
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="controlnets" className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-slate-700/50">
            <TabsTrigger value="controlnets" className="transition-all duration-300">
              ControlNets
            </TabsTrigger>
            <TabsTrigger value="unions" className="transition-all duration-300">
              ControlNet Unions
            </TabsTrigger>
          </TabsList>

          <TabsContent value="controlnets" className="space-y-4 mt-4 animate-in fade-in-0 duration-500">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="text-white font-medium">ControlNets</h3>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-slate-500" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Add ControlNets with optional LoRA enhancements</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select onValueChange={(value) => addControlNet(value)}>
                <SelectTrigger className="w-48 bg-slate-700 border-slate-600 hover:bg-slate-600 transition-colors duration-200">
                  <SelectValue placeholder="Add ControlNet" />
                </SelectTrigger>
                <SelectContent>
                  {controlTypes.map(type => (
                    <SelectItem key={type} value={type}>
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              {controlNets.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <Image className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p className="text-lg">No ControlNets added</p>
                  <p className="text-sm">Add a ControlNet to get started with image control</p>
                </div>
              ) : (
                controlNets.map((controlNet) => (
                  <ControlNetComponent key={controlNet.id} controlNet={controlNet} />
                ))
              )}
            </div>
          </TabsContent>

          <TabsContent value="unions" className="space-y-4 mt-4 animate-in fade-in-0 duration-500">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="text-white font-medium">ControlNet Unions</h3>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="w-4 h-4 text-slate-500" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Combine multiple ControlNets for complex control</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Button 
                onClick={addControlUnion}
                className="bg-purple-600 hover:bg-purple-700 transition-all duration-200 hover:scale-105"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Union
              </Button>
            </div>

            <div className="space-y-3">
              {controlUnions.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <Grid className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p className="text-lg">No ControlNet Unions created</p>
                  <p className="text-sm">Create unions to combine multiple ControlNets</p>
                </div>
              ) : (
                controlUnions.map((union) => (
                  <div key={union.id} className="p-4 bg-slate-900/30 rounded-lg border border-slate-600 animate-in slide-in-from-left-2 duration-500">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-white font-medium">{union.name}</h4>
                      <Badge variant="secondary" className="bg-purple-500/20 text-purple-300">
                        {union.mode}
                      </Badge>
                    </div>
                    <p className="text-slate-400 text-sm">
                      Union configuration interface coming soon...
                    </p>
                  </div>
                ))
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default ControlSection;
