# =============================================================================
# __init__.py
# =============================================================================

"""
ComfyUI Model Optimizer Package
Provides Flash Attention and torch.compile optimization nodes for ComfyUI models.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Package metadata
__version__ = "1.0.0"
__author__ = "ComfyUI Model Optimizer"
__description__ = "Flash Attention and torch.compile optimization nodes for ComfyUI"
