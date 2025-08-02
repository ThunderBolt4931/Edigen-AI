import os
import json
import torch
import folder_paths
from safetensors.torch import save_file, load_file
from safetensors import safe_open

# --- Helper Functions for deep tensor handling ---

def _sanitize_and_extract_tensors(d, prefix, tensors_dict):
    """
    Recursively scans a dictionary, extracts tensors, and replaces them with placeholders.
    """
    sanitized_dict = {}
    if not isinstance(d, dict):
        return d
        
    for k, v in d.items():
        # Create a unique key for the current position
        current_prefix = f"{prefix}_{k}"
        if isinstance(v, torch.Tensor):
            # This is a tensor, so we extract it
            tensors_dict[current_prefix] = v
            # And replace it with a placeholder
            sanitized_dict[k] = {"__tensor_ref__": current_prefix}
        elif isinstance(v, dict):
            # This is a nested dictionary, recurse into it
            sanitized_dict[k] = _sanitize_and_extract_tensors(v, current_prefix, tensors_dict)
        else:
            # This is a JSON-serializable type, keep it as is
            sanitized_dict[k] = v
    return sanitized_dict

def _reconstruct_tensors_in_dict(d, tensors_dict):
    """
    Recursively scans a dictionary, finds placeholders, and replaces them with tensors.
    """
    if not isinstance(d, dict):
        return d

    # Check if the dictionary is a placeholder itself
    if "__tensor_ref__" in d:
        tensor_key = d["__tensor_ref__"]
        if tensor_key in tensors_dict:
            return tensors_dict[tensor_key]
        else:
            raise ValueError(f"Tensor reference '{tensor_key}' not found in loaded tensors.")

    # Otherwise, iterate over its items and reconstruct
    reconstructed_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            reconstructed_dict[k] = _reconstruct_tensors_in_dict(v, tensors_dict)
        else:
            reconstructed_dict[k] = v
    return reconstructed_dict


# --- Node Classes ---

# Create a dedicated folder for the cached conditioning inside the ComfyUI/output directory
CACHE_DIR = os.path.join(folder_paths.get_output_directory(), "conditioning_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Register this new directory with ComfyUI's folder_paths so it can be used for loading
folder_paths.add_model_folder_path("conditioning_cache", CACHE_DIR)


class SaveConditioningNode:
    """
    Saves conditioning and optional pooled_output tensors to a .safetensors file.
    """
    OUTPUT_NODE = True

    def __init__(self):
        print("SaveConditioningNode Initialized")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "filename_prefix": ("STRING", {"default": "prompt_conditioning"}),
            },
            "optional": {
                "pooled_output": ("POOLED_OUTPUT",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_conditioning"
    CATEGORY = "caching"

    def save_conditioning(self, conditioning, filename_prefix, pooled_output=None):
        filename = f"{filename_prefix}.safetensors"
        file_path = os.path.join(CACHE_DIR, filename)

        tensors_to_save = {}
        sanitized_dicts = []

        # Unpack the conditioning list
        for i, (tensor, dictionary) in enumerate(conditioning):
            # Save the main tensor of the conditioning pair
            tensors_to_save[f"cond_tensor_{i}"] = tensor
            # Sanitize the accompanying dictionary, extracting any nested tensors
            sanitized_dict = _sanitize_and_extract_tensors(dictionary, f"cond_dict_{i}", tensors_to_save)
            sanitized_dicts.append(sanitized_dict)
        
        # Handle optional top-level pooled_output
        if pooled_output is not None:
            tensors_to_save["pooled_output"] = pooled_output

        # Convert the list of sanitized dictionaries to a JSON string for metadata
        metadata = {"conditioning_dicts": json.dumps(sanitized_dicts)}

        save_file(tensors_to_save, file_path, metadata=metadata)

        print(f"Conditioning saved to: {file_path}")
        return {"ui": {"text": [f"Saved to {filename}"]}}


class LoadConditioningNode:
    """
    Loads conditioning and pooled_output tensors from a .safetensors file
    and reconstructs the original data structure.
    """
    def __init__(self):
        print("LoadConditioningNode Initialized")

    @classmethod
    def INPUT_TYPES(cls):
        try:
            file_list = [f for f in os.listdir(CACHE_DIR) if f.endswith(".safetensors")]
        except FileNotFoundError:
            file_list = []
        return {
            "required": {
                "filename": (file_list, )
            }
        }

    RETURN_TYPES = ("CONDITIONING", "POOLED_OUTPUT")
    RETURN_NAMES = ("conditioning", "pooled_output")
    FUNCTION = "load_conditioning"
    CATEGORY = "caching"

    def load_conditioning(self, filename):
        file_path = os.path.join(CACHE_DIR, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Conditioning file not found: {file_path}")

        loaded_tensors = load_file(file_path)
        
        metadata = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}

        if "conditioning_dicts" not in metadata:
            raise ValueError(f"Metadata for conditioning structure not found in {filename}")

        sanitized_dicts = json.loads(metadata["conditioning_dicts"])
        reconstructed_conditioning = []

        # Re-assemble the conditioning list
        for i, sanitized_dict in enumerate(sanitized_dicts):
            tensor_key = f"cond_tensor_{i}"
            if tensor_key not in loaded_tensors:
                raise ValueError(f"Main conditioning tensor '{tensor_key}' not found in {filename}")
            
            main_tensor = loaded_tensors[tensor_key]
            # Reconstruct the dictionary by re-inserting nested tensors
            reconstructed_dict = _reconstruct_tensors_in_dict(sanitized_dict, loaded_tensors)
            reconstructed_conditioning.append((main_tensor, reconstructed_dict))

        # Get the optional top-level pooled_output
        pooled_output = loaded_tensors.get("pooled_output", None)
            
        print(f"Loaded conditioning from: {filename}")
        return (reconstructed_conditioning, pooled_output)
