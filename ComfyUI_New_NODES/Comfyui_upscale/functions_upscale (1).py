#---------------------------------------------------------------------------------------------------------------------#
# Comfyroll Studio custom nodes by RockOfFire and Akatsuzi    https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes                             
# for ComfyUI                                                 https://github.com/comfyanonymous/ComfyUI                                               
#---------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------#
# UPSCALE FUNCTIONS
#---------------------------------------------------------------------------------------------------------------------#
# These functions are based on WAS nodes Image Resize and the Comfy Extras upscale with model nodes

import torch
#import os
from comfy_extras.chainner_models import model_loading
from comfy import model_management
import numpy as np
import comfy.utils
import folder_paths
from PIL import Image

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def load_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
    out = model_loading.load_state_dict(sd).eval()
    return out
    
def upscale_with_model(upscale_model, image):
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)
    free_memory = model_management.get_free_memory(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()
    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    return s        

# In functions_upscale.py

def apply_resize_image(image: Image.Image, original_width, original_height, rounding_modulus, mode='scale', supersample='true', factor: int = 2, width: int = 1024, height: int = 1024, resample='bicubic'): 

    # Define the maximum dimension for the longest side.
    MAX_DIMENSION = 1080
    
    current_width, current_height = image.size

    # If the image's longest side is already 1080, no resize is needed.
    if max(current_width, current_height) == MAX_DIMENSION:
        return image
        
    # Determine target dimensions to enforce 1080p limit, preserving aspect ratio.
    # This logic overrides the user's 'factor' and 'width' inputs.
    aspect_ratio = current_width / current_height

    if current_width >= current_height:
        # For landscape or square images, set width to 1080
        new_width = MAX_DIMENSION
        new_height = int(new_width / aspect_ratio)
    else:
        # For portrait images, set height to 1080
        new_height = MAX_DIMENSION
        new_width = int(new_height * aspect_ratio)

    # Define a dictionary of resampling filters
    resample_filters = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'lanczos': 1}
    
    # Apply supersample for better quality on upscaling
    if supersample == 'true' and (new_width > current_width or new_height > current_height):
        # Supersampling is beneficial when enlarging the image
        image = image.resize((new_width * 4, new_height * 4), resample=Image.Resampling(resample_filters[resample]))

    # Resize the image to the final calculated 1080p dimensions
    print(f"[Info] CR Upscale: Resizing image to {new_width}x{new_height}")
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))
    
    return resized_image

