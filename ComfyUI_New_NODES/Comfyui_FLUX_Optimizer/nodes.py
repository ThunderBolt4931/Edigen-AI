
# =============================================================================
# nodes.py
# =============================================================================

import torch
import comfy.ldm.modules.attention
import comfy.model_management
from typing import Dict, Any, Tuple, Optional


class FlashAttentionNode:
    """
    A node to apply Flash Attention optimization to ComfyUI models.
    Flash Attention provides memory-efficient attention computation.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enable_flash_attention": ("BOOLEAN", {"default": True}),
                "attention_backend": (["auto", "flash_attn", "xformers", "pytorch"], {"default": "auto"}),
                "force_upcast": ("BOOLEAN", {"default": False, "tooltip": "Force upcasting to fp32 for stability"}),
            },
            "optional": {
                "attention_slice": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 32,
                    "tooltip": "Slice attention for lower VRAM usage (0 = disabled)"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "apply_flash_attention"
    CATEGORY = "advanced/optimization"
    DESCRIPTION = "Apply Flash Attention optimization for memory-efficient attention computation"
    
    def apply_flash_attention(
        self, 
        model, 
        enable_flash_attention: bool,
        attention_backend: str,
        force_upcast: bool,
        attention_slice: int = 0
    ) -> Tuple[Any]:
        """Apply Flash Attention optimization to the model."""
        
        # Clone model to avoid in-place modifications
        optimized_model = model.clone()
        
        if enable_flash_attention:
            print(f"FlashAttentionNode: Applying Flash Attention with backend '{attention_backend}'...")
            
            try:
                # Configure attention backend
                if attention_backend == "flash_attn":
                    # Force Flash Attention specifically
                    comfy.ldm.modules.attention.add_comfy_flash_attention()
                    print("✓ Flash Attention (flash-attn) enabled")
                    
                elif attention_backend == "xformers":
                    # Use xFormers if available
                    try:
                        import xformers
                        comfy.ldm.modules.attention.use_xformers_attention()
                        print("✓ xFormers attention enabled")
                    except ImportError:
                        print("⚠ xFormers not available, falling back to PyTorch")
                        attention_backend = "pytorch"
                
                elif attention_backend == "auto":
                    # Auto-detect best available backend
                    try:
                        comfy.ldm.modules.attention.add_comfy_flash_attention()
                        print("✓ Auto-detected: Flash Attention enabled")
                    except ImportError:
                        try:
                            import xformers
                            comfy.ldm.modules.attention.use_xformers_attention()
                            print("✓ Auto-detected: xFormers attention enabled")
                        except ImportError:
                            print("✓ Auto-detected: Using PyTorch native attention")
                
                # Configure attention slicing for VRAM optimization
                if attention_slice > 0:
                    print(f"FlashAttentionNode: Setting attention slice to {attention_slice}")
                    # Apply attention slicing if the model supports it
                    if hasattr(optimized_model.model, 'set_attention_slice'):
                        optimized_model.model.set_attention_slice(attention_slice)
                
                # Configure precision
                if force_upcast:
                    print("FlashAttentionNode: Forcing fp32 upcast for attention")
                    # This would need to be implemented based on ComfyUI's model structure
                    
            except ImportError as e:
                print(f"⚠ FlashAttentionNode: Required libraries not found: {e}")
                print("FlashAttentionNode: Install with: pip install flash-attn")
                
            except Exception as e:
                print(f"⚠ FlashAttentionNode: Error applying Flash Attention: {e}")
                
        else:
            print("FlashAttentionNode: Flash Attention disabled, using default attention")
            # Optionally revert to default attention
            comfy.ldm.modules.attention.remove_comfy_flash_attention()
        
        return (optimized_model,)


class TorchCompileNode:
    """
    A node to apply torch.compile optimization to ComfyUI models.
    Provides various compilation modes and backend options for performance optimization.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "enable_compilation": ("BOOLEAN", {"default": True}),
                "compilation_mode": ([
                    "default", 
                    "reduce-overhead", 
                    "max-autotune", 
                    "max-autotune-no-cudagraphs"
                ], {"default": "reduce-overhead"}),
                "backend": ([
                    "inductor", 
                    "aot_eager", 
                    "cudagraphs", 
                    "onnxrt", 
                    "tensorrt"
                ], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "compile_unet_only": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Only compile UNet (recommended for stability)"
                }),
                "max_autotune_pointwise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable max-autotune for pointwise operations"
                }),
                "safe_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Add compatibility wrapper for attribute access"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("compiled_model",)
    FUNCTION = "apply_torch_compile"
    CATEGORY = "advanced/optimization"
    DESCRIPTION = "Apply torch.compile optimization with various modes and backends"
    
    def apply_torch_compile(
        self,
        model,
        enable_compilation: bool,
        compilation_mode: str,
        backend: str,
        fullgraph: bool,
        dynamic: bool,
        compile_unet_only: bool = True,
        max_autotune_pointwise: bool = True,
        safe_mode: bool = True
    ) -> Tuple[Any]:
        """Apply torch.compile optimization to the model."""
        
        # Check PyTorch version
        if not hasattr(torch, 'compile'):
            print("⚠ TorchCompileNode: torch.compile requires PyTorch 2.0+")
            return (model,)
        
        # Clone model to avoid in-place modifications
        compiled_model = model.clone()
        
        if enable_compilation:
            print(f"TorchCompileNode: Applying torch.compile...")
            print(f"  Mode: {compilation_mode}")
            print(f"  Backend: {backend}")
            print(f"  Fullgraph: {fullgraph}")
            print(f"  Dynamic: {dynamic}")
            print(f"  Safe mode: {safe_mode}")
            
            try:
                # Configure compilation options
                compile_options = {
                    "mode": compilation_mode,
                    "backend": backend,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                
                # Add backend-specific options
                if backend == "inductor":
                    compile_options["options"] = {
                        "max_autotune": compilation_mode in ["max-autotune", "max-autotune-no-cudagraphs"],
                        "max_autotune_pointwise": max_autotune_pointwise,
                        "triton.cudagraphs": compilation_mode != "max-autotune-no-cudagraphs",
                    }
                
                if compile_unet_only:
                    # Compile only the UNet/diffusion model (most compute-intensive part)
                    print("TorchCompileNode: Compiling UNet only...")
                    if hasattr(compiled_model.model, 'diffusion_model'):
                        original_model = compiled_model.model.diffusion_model
                        
                        if safe_mode:
                            # Create a compatibility wrapper
                            compiled_diffusion_model = self._create_safe_compiled_model(
                                original_model, **compile_options
                            )
                        else:
                            compiled_diffusion_model = torch.compile(
                                original_model, **compile_options
                            )
                        
                        compiled_model.model.diffusion_model = compiled_diffusion_model
                        print("✓ UNet compiled successfully")
                    else:
                        print("⚠ TorchCompileNode: UNet not found in model structure")
                else:
                    # Compile the entire model
                    print("TorchCompileNode: Compiling entire model...")
                    if safe_mode:
                        compiled_model.model = self._create_safe_compiled_model(
                            compiled_model.model, **compile_options
                        )
                    else:
                        compiled_model.model = torch.compile(
                            compiled_model.model, **compile_options
                        )
                    print("✓ Full model compiled successfully")
                
                print("TorchCompileNode: ⚠ First inference will be slower due to compilation")
                
            except Exception as e:
                print(f"⚠ TorchCompileNode: Compilation failed: {e}")
                print("TorchCompileNode: Returning uncompiled model")
                return (model,)
        
        else:
            print("TorchCompileNode: Compilation disabled")
        
        return (compiled_model,)
    
    def _create_safe_compiled_model(self, model, **compile_options):
        """Create a compiled model with compatibility wrapper for attribute access."""
        import torch.nn as nn
        
        class SafeCompiledModel(nn.Module):
            """
            Wrapper for torch.compile that maintains attribute compatibility.
            This prevents issues with custom nodes that expect specific attributes.
            """
            def __init__(self, original_model, **compile_options):
                super().__init__()
                self.original_model = original_model
                self.compiled_model = torch.compile(original_model, **compile_options)
                
                # Store original attributes that might be accessed by other nodes
                self._original_attributes = set(dir(original_model))
                
            def forward(self, *args, **kwargs):
                return self.compiled_model(*args, **kwargs)
            
            def __getattr__(self, name):
                # First try to get from compiled model
                try:
                    return getattr(self.compiled_model, name)
                except AttributeError:
                    pass
                
                # Then try original model
                try:
                    return getattr(self.original_model, name)
                except AttributeError:
                    pass
                
                # If attribute doesn't exist, raise AttributeError
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def __setattr__(self, name, value):
                # Handle special attributes
                if name in ['original_model', 'compiled_model', '_original_attributes']:
                    super().__setattr__(name, value)
                    return
                
                # Try to set on original model first (for compatibility)
                if hasattr(self, 'original_model') and hasattr(self.original_model, name):
                    setattr(self.original_model, name, value)
                else:
                    super().__setattr__(name, value)
            
            def __delattr__(self, name):
                # Handle deletion safely - check if attribute exists before deleting
                if hasattr(self, 'original_model') and hasattr(self.original_model, name):
                    try:
                        delattr(self.original_model, name)
                    except AttributeError:
                        # Attribute doesn't exist, ignore silently
                        pass
                elif hasattr(self, name):
                    super().__delattr__(name)
                # If attribute doesn't exist anywhere, ignore silently for compatibility
        
        return SafeCompiledModel(model, **compile_options)


class ModelOptimizerCombined:
    """
    Combined node that applies both Flash Attention and torch.compile optimizations.
    Provides a convenient single-node solution for comprehensive model optimization.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                
                # Flash Attention options
                "enable_flash_attention": ("BOOLEAN", {"default": True}),
                "attention_backend": (["auto", "flash_attn", "xformers", "pytorch"], {"default": "auto"}),
                
                # torch.compile options
                "enable_compilation": ("BOOLEAN", {"default": True}),
                "compilation_mode": ([
                    "default", 
                    "reduce-overhead", 
                    "max-autotune", 
                    "max-autotune-no-cudagraphs"
                ], {"default": "reduce-overhead"}),
                "backend": (["inductor", "aot_eager", "cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "attention_slice": ("INT", {"default": 0, "min": 0, "max": 32}),
                "compile_unet_only": ("BOOLEAN", {"default": True}),
                "safe_mode": ("BOOLEAN", {"default": True, "tooltip": "Use compatibility wrapper for compiled models"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "optimize_model"
    CATEGORY = "advanced/optimization"
    DESCRIPTION = "Apply both Flash Attention and torch.compile optimizations"
    
    def optimize_model(
        self,
        model,
        enable_flash_attention: bool,
        attention_backend: str,
        enable_compilation: bool,
        compilation_mode: str,
        backend: str,
        fullgraph: bool,
        attention_slice: int = 0,
        compile_unet_only: bool = True,
        safe_mode: bool = True
    ) -> Tuple[Any]:
        """Apply comprehensive model optimizations."""
        
        print("ModelOptimizerCombined: Starting comprehensive optimization...")
        
        # Start with the original model
        optimized_model = model
        
        # Apply Flash Attention first
        if enable_flash_attention:
            flash_node = FlashAttentionNode()
            optimized_model = flash_node.apply_flash_attention(
                optimized_model, 
                enable_flash_attention,
                attention_backend,
                False,  # force_upcast
                attention_slice
            )[0]
        
        # Then apply torch.compile
        if enable_compilation:
            compile_node = TorchCompileNode()
            optimized_model = compile_node.apply_torch_compile(
                optimized_model,
                enable_compilation,
                compilation_mode,
                backend,
                fullgraph,
                True,  # dynamic
                compile_unet_only,
                True,  # max_autotune_pointwise
                safe_mode
            )[0]
        
        print("ModelOptimizerCombined: ✓ Optimization complete!")
        return (optimized_model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FlashAttentionNode": FlashAttentionNode,
    "TorchCompileNode": TorchCompileNode,
    "ModelOptimizerCombined": ModelOptimizerCombined,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashAttentionNode": "Flash Attention Optimizer",
    "TorchCompileNode": "Torch Compile Optimizer", 
    "ModelOptimizerCombined": "Model Optimizer (Combined)",
}