
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BookOpen, Layers, Settings, Grid, Zap, Code, Image, Sliders } from "lucide-react";

const DocumentationPanel = () => {
  return (
    <div className="space-y-6">
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            AI Model Control Documentation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="loras" className="w-full">
            <TabsList className="grid w-full grid-cols-4 bg-slate-700/50">
              <TabsTrigger value="loras">LoRAs</TabsTrigger>
              <TabsTrigger value="control-loras">Control LoRAs</TabsTrigger>
              <TabsTrigger value="controlnets">ControlNets</TabsTrigger>
              <TabsTrigger value="unions">ControlNet Unions</TabsTrigger>
            </TabsList>

            <TabsContent value="loras" className="space-y-6 mt-6">
              <div>
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Layers className="w-6 h-6" />
                  LoRAs (Low-Rank Adaptation)
                </h3>
                
                <div className="space-y-4">
                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">What are LoRAs?</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <p className="text-slate-300">
                        LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large language models and diffusion models by updating only a small subset of parameters.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Advantages:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Minimal memory footprint</li>
                            <li>• Fast training and inference</li>
                            <li>• Easy to share and combine</li>
                            <li>• Preserves base model quality</li>
                          </ul>
                        </div>
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Use Cases:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Style adaptation</li>
                            <li>• Character consistency</li>
                            <li>• Artistic techniques</li>
                            <li>• Concept injection</li>
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white flex items-center gap-2">
                        <Sliders className="w-5 h-5" />
                        LoRA Parameters
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-3">
                          <div>
                            <Badge className="mb-2 bg-purple-500/20 text-purple-300">Scale</Badge>
                            <p className="text-sm text-slate-300">
                              Controls the strength of the LoRA effect. Range: -2.0 to 2.0
                            </p>
                            <ul className="text-xs text-slate-400 mt-1">
                              <li>• 0.0: No effect</li>
                              <li>• 1.0: Full strength (recommended)</li>
                              <li>• &gt;1.0: Amplified effect</li>
                              <li>• &lt;0.0: Negative/inverse effect</li>
                            </ul>
                          </div>
                        </div>
                        <div className="space-y-3">
                          <div>
                            <Badge className="mb-2 bg-blue-500/20 text-blue-300">Path/ID</Badge>
                            <p className="text-sm text-slate-300">
                              Reference to the LoRA model file or identifier
                            </p>
                            <ul className="text-xs text-slate-400 mt-1">
                              <li>• File path: ./models/lora_name.safetensors</li>
                              <li>• HuggingFace ID: username/model-name</li>
                              <li>• Civitai ID: model_id:version_id</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">Implementation Logic</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-slate-800 p-4 rounded-lg">
                        <code className="text-sm text-green-400">
                          <pre>{`// LoRA Application Process
1. Load base model weights
2. Load LoRA adaptation matrices (A, B)
3. Compute LoRA weights: W_lora = scale * (A @ B)
4. Apply to model: W_final = W_base + W_lora
5. Forward pass with adapted weights

// Multiple LoRAs
for lora in loras:
    W_final += lora.scale * (lora.A @ lora.B)`}</pre>
                        </code>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="control-loras" className="space-y-6 mt-6">
              <div>
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Settings className="w-6 h-6" />
                  Control LoRAs
                </h3>
                
                <div className="space-y-4">
                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">What are Control LoRAs?</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <p className="text-slate-300">
                        Control LoRAs are specialized LoRA adapters designed to provide structural and compositional control over image generation while maintaining the efficiency of the LoRA approach.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Key Features:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Lightweight control mechanism</li>
                            <li>• Compatible with base LoRAs</li>
                            <li>• Step-wise control timing</li>
                            <li>• Adjustable influence strength</li>
                          </ul>
                        </div>
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Control Types:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Edge detection (Canny)</li>
                            <li>• Depth mapping</li>
                            <li>• Pose estimation</li>
                            <li>• Normal mapping</li>
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">Step Control Parameters</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <Badge className="mb-2 bg-green-500/20 text-green-300">Start Step</Badge>
                          <p className="text-sm text-slate-300">
                            Denoising step to begin applying control
                          </p>
                        </div>
                        <div>
                          <Badge className="mb-2 bg-red-500/20 text-red-300">End Step</Badge>
                          <p className="text-sm text-slate-300">
                            Denoising step to stop applying control
                          </p>
                        </div>
                        <div>
                          <Badge className="mb-2 bg-yellow-500/20 text-yellow-300">Strength</Badge>
                          <p className="text-sm text-slate-300">
                            Intensity of control influence (0.0-2.0)
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="controlnets" className="space-y-6 mt-6">
              <div>
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Grid className="w-6 h-6" />
                  ControlNets
                </h3>
                
                <div className="space-y-4">
                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">What are ControlNets?</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <p className="text-slate-300">
                        ControlNet is a neural network architecture that allows precise control over diffusion models by adding spatial conditioning to the generation process.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Architecture:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Parallel neural network branch</li>
                            <li>• Zero-initialized connections</li>
                            <li>• Preserves original model weights</li>
                            <li>• Adds spatial conditioning</li>
                          </ul>
                        </div>
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Input Types:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Control images (edges, depth, etc.)</li>
                            <li>• Preprocessed conditions</li>
                            <li>• Mask-based guidance</li>
                            <li>• Multi-modal inputs</li>
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">ControlNet Types</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          { name: 'Canny', desc: 'Edge detection control', color: 'blue' },
                          { name: 'Depth', desc: 'Depth map guidance', color: 'green' },
                          { name: 'Pose', desc: 'Human pose control', color: 'purple' },
                          { name: 'Normal', desc: 'Surface normal maps', color: 'orange' },
                          { name: 'Semantic', desc: 'Segmentation masks', color: 'pink' },
                          { name: 'Scribble', desc: 'Hand-drawn sketches', color: 'yellow' },
                          { name: 'LineArt', desc: 'Clean line drawings', color: 'cyan' },
                          { name: 'MLSD', desc: 'Line segment detection', color: 'red' },
                          { name: 'Tile', desc: 'Texture tiling control', color: 'indigo' }
                        ].map((type) => (
                          <div key={type.name} className="p-3 bg-slate-800 rounded-lg border border-slate-600">
                            <Badge className={`mb-2 bg-${type.color}-500/20 text-${type.color}-300`}>
                              {type.name}
                            </Badge>
                            <p className="text-sm text-slate-300">{type.desc}</p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">Implementation Architecture</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="bg-slate-800 p-4 rounded-lg">
                        <code className="text-sm text-green-400">
                          <pre>{`// ControlNet Forward Pass
class ControlNet:
    def forward(self, x, control_input, timestep):
        # Process control input
        control_features = self.control_net(control_input)
        
        # Original UNet forward pass
        unet_features = self.unet(x, timestep)
        
        # Add control guidance
        controlled_features = unet_features + control_features
        
        return controlled_features

// Multiple ControlNets
for controlnet in controlnets:
    if current_step >= controlnet.start_step and current_step <= controlnet.end_step:
        control_signal += controlnet.strength * controlnet(control_image)`}</pre>
                        </code>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="unions" className="space-y-6 mt-6">
              <div>
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Zap className="w-6 h-6" />
                  ControlNet Unions
                </h3>
                
                <div className="space-y-4">
                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">What are ControlNet Unions?</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <p className="text-slate-300">
                        ControlNet Unions allow combining multiple ControlNet models into a single, efficient architecture that can handle multiple types of control simultaneously.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Benefits:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Reduced memory usage</li>
                            <li>• Faster inference</li>
                            <li>• Better feature integration</li>
                            <li>• Unified control interface</li>
                          </ul>
                        </div>
                        <div className="space-y-2">
                          <h4 className="font-semibold text-white">Union Modes:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• More Prompt: Emphasize text guidance</li>
                            <li>• More Control: Emphasize structural control</li>
                            <li>• Balanced: Equal weight distribution</li>
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">Union Architecture</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="bg-slate-800 p-4 rounded-lg mb-4">
                        <code className="text-sm text-green-400">
                          <pre>{`// ControlNet Union Implementation
class ControlNetUnion:
    def __init__(self, control_types, union_mode):
        self.control_types = control_types
        self.union_mode = union_mode
        self.shared_encoder = SharedEncoder()
        self.control_heads = {
            ctype: ControlHead(ctype) for ctype in control_types
        }
    
    def forward(self, inputs, mode_weights):
        # Shared feature extraction
        shared_features = self.shared_encoder(inputs['image'])
        
        # Control-specific processing
        control_outputs = {}
        for ctype, control_input in inputs.items():
            if ctype in self.control_heads:
                control_outputs[ctype] = self.control_heads[ctype](
                    shared_features, control_input
                )
        
        # Weighted combination based on union mode
        final_output = self.combine_controls(
            control_outputs, mode_weights
        )
        
        return final_output`}</pre>
                        </code>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="p-3 bg-slate-800 rounded-lg">
                          <Badge className="mb-2 bg-green-500/20 text-green-300">More Prompt</Badge>
                          <p className="text-sm text-slate-300">
                            Weights: Text 0.7, Control 0.3
                          </p>
                        </div>
                        <div className="p-3 bg-slate-800 rounded-lg">
                          <Badge className="mb-2 bg-blue-500/20 text-blue-300">Balanced</Badge>
                          <p className="text-sm text-slate-300">
                            Weights: Text 0.5, Control 0.5
                          </p>
                        </div>
                        <div className="p-3 bg-slate-800 rounded-lg">
                          <Badge className="mb-2 bg-purple-500/20 text-purple-300">More Control</Badge>
                          <p className="text-sm text-slate-300">
                            Weights: Text 0.3, Control 0.7
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-900/50 border-slate-600">
                    <CardHeader>
                      <CardTitle className="text-lg text-white">Best Practices</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-3">
                          <h4 className="font-semibold text-white">Combination Strategy:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Start with 2-3 compatible controls</li>
                            <li>• Test different union modes</li>
                            <li>• Adjust individual strengths</li>
                            <li>• Monitor for conflicts</li>
                          </ul>
                        </div>
                        <div className="space-y-3">
                          <h4 className="font-semibold text-white">Common Combinations:</h4>
                          <ul className="text-sm text-slate-300 space-y-1">
                            <li>• Canny + Depth (structure)</li>
                            <li>• Pose + Semantic (people)</li>
                            <li>• Normal + Tile (surfaces)</li>
                            <li>• LineArt + Scribble (sketches)</li>
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default DocumentationPanel;
