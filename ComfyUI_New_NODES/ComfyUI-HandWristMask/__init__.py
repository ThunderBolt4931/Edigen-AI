# __init__.py for the Hand Wrist Mask custom node

# Import the node class from the other file
from .hand_wrist_mask import HandWristMask

# A dictionary that maps the node's class name to the class itself.
# This is used by ComfyUI to load the node.
NODE_CLASS_MAPPINGS = {
    "HandWristMask": HandWristMask
}

# A dictionary that maps the node's class name to a more user-friendly
# display name that will appear in the ComfyUI interface.
NODE_DISPLAY_NAME_MAPPINGS = {
    "HandWristMask": "Hand Wrist Mask Creator"
}

# A tuple that tells ComfyUI which dictionaries to look at.
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ ComfyUI-HandWristMask: Custom node loaded successfully.")
