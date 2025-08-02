# In file: nodes_upscale.py

import torch
import numpy as np
import folder_paths
from PIL import Image
from ..categories import icons
from .functions_upscale import *

#---------------------------------------------------------------------------------------------------------------------#
# NODES
#---------------------------------------------------------------------------------------------------------------------#

class CR_UpscaleImage:

    @classmethod
    def INPUT_TYPES(s):
        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]
       
        return {"required": {
            "image": ("IMAGE",),
            "upscale_model": (folder_paths.get_filename_list("upscale_models"), ),
            "resampling_method": (resampling_methods,),                     
            "supersample": (["true", "false"],),   
        }}

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("IMAGE", "show_help", )
    FUNCTION = "upscale"
    CATEGORY = icons.get("Comfyroll/Upscale")
    
    def upscale(self, image, upscale_model, supersample='true', resampling_method="lanczos"):
        # Load upscale model 
        up_model = load_model(upscale_model)

        # Upscale with model
        up_image = upscale_with_model(up_model, image)

        pil_img_orig = tensor2pil(image[0])
        original_width, original_height = pil_img_orig.size

        # Image resize
        scaled_images = []
        
        # The modified apply_resize_image now handles all scaling logic to 1080p.
        # We pass placeholder values for the unused parameters.
        for img in up_image:
            scaled_images.append(pil2tensor(apply_resize_image(
                tensor2pil(img), 
                original_width, 
                original_height, 
                rounding_modulus=8, # Placeholder
                mode='rescale', # Placeholder
                supersample=supersample, 
                factor=1, # Placeholder
                width=1080, # Placeholder
                resample=resampling_method
            )))
        images_out = torch.cat(scaled_images, dim=0)
 
        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Upscale-Nodes#cr-upscale-image"
        return (images_out, show_help, )        
 
#---------------------------------------------------------------------------------------------------------------------
class CR_MultiUpscaleStack:

    @classmethod
    def INPUT_TYPES(s):
        up_models = ["None"] + folder_paths.get_filename_list("upscale_models")
        
        return {"required": {
            "switch_1": (["On","Off"],),              
            "upscale_model_1": (up_models, ),
            "switch_2": (["On","Off"],),                          
            "upscale_model_2": (up_models, ),
            "switch_3": (["On","Off"],),                        
            "upscale_model_3": (up_models, ),
        },
        "optional": {"upscale_stack": ("UPSCALE_STACK",),}}

    RETURN_TYPES = ("UPSCALE_STACK", "STRING", )
    RETURN_NAMES = ("UPSCALE_STACK", "show_help", )
    FUNCTION = "stack"
    CATEGORY = icons.get("Comfyroll/Upscale")
    
    def stack(self, switch_1, upscale_model_1, switch_2, upscale_model_2, switch_3, upscale_model_3, upscale_stack=None):
        # Initialise the list
        upscale_list = list()
        
        if upscale_stack is not None:
            upscale_list.extend([model for model in upscale_stack if model != "None"])
        
        # Rescale factors are removed as they are no longer used.
        if upscale_model_1 != "None" and switch_1 == "On":
            upscale_list.append(upscale_model_1)

        if upscale_model_2 != "None" and switch_2 == "On":
            upscale_list.append(upscale_model_2)

        if upscale_model_3 != "None" and switch_3 == "On":
            upscale_list.append(upscale_model_3)

        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Upscale-Nodes#cr-multi-upscale-stack"
        return (upscale_list, show_help, )

#---------------------------------------------------------------------------------------------------------------------
class CR_ApplyMultiUpscale:

    @classmethod
    def INPUT_TYPES(s):
        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]
        
        return {"required": {
            "image": ("IMAGE",),
            "resampling_method": (resampling_methods,),
            "supersample": (["true", "false"],),                     
            "upscale_stack": ("UPSCALE_STACK",),
        }}
    
    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("IMAGE", "show_help", )
    FUNCTION = "apply"
    CATEGORY = icons.get("Comfyroll/Upscale")

    def apply(self, image, resampling_method, supersample, upscale_stack):
        pil_img = tensor2pil(image[0])
        original_width, original_height = pil_img.size
        
        temp_image = image
        
        # Loop through the list of models
        for upscale_model in upscale_stack:
            print(f"[Info] CR Apply Multi Upscale: Applying model {upscale_model}")
            up_model = load_model(upscale_model)
            temp_image = upscale_with_model(up_model, temp_image)
            
        # After all models are applied, do the final resize to 1080p
        scaled_images = []
        for img in temp_image:
            scaled_images.append(pil2tensor(apply_resize_image(
                tensor2pil(img), 
                original_width, 
                original_height, 
                rounding_modulus=8, # Placeholder
                mode='rescale', # Placeholder
                supersample=supersample, 
                factor=1, # Placeholder
                width=1080, # Placeholder
                resample=resampling_method
            )))
        final_image = torch.cat(scaled_images, dim=0)
            
        show_help = "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Upscale-Nodes#cr-apply-multi-upscale"
        return (final_image, show_help, )