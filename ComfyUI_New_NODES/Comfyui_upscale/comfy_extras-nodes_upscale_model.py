import logging
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import comfy.model_management
import torch
import comfy.utils
import folder_paths

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("Successfully imported spandrel_extra_arches: support for non commercial upscale models.")
except:
    pass

class UpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        return (out, )


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        MAX_DIMENSION = 2160
        h, w = image.shape[1:3]

        # --- ADDED: Early return if image already exceeds the max dimension ---
        if max(h, w) > MAX_DIMENSION:
            print(f"[ImageUpscaleWithModel] Image size ({w}x{h}) is larger than {MAX_DIMENSION}px. Skipping upscale and returning original image.")
            return (image,)
        # --- END OF ADDITION ---

        # Calculate the final scale factor, ensuring it doesn't exceed the limit
        final_scale = 1.0
        if h > 0 and w > 0:
            scale_limit = MAX_DIMENSION / max(h, w)
            final_scale = min(upscale_model.scale, scale_limit)
        
        if final_scale < 1.0:
            final_scale = 1.0
            
        # If no upscaling is needed, return the original image
        if final_scale == 1.0:
            return (image,)

        device = model_management.get_torch_device()
        
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(final_scale, 1.0) * 384.0
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        # Dynamically set tile size to be no larger than the image's smallest dimension
        tile = min(h, w, 512)
        print(f"[ImageUpscaleWithModel] DEBUG: Upscaling image from {w}x{h}. Tile size: {tile}. Final scale: {final_scale:.2f}")
        
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=final_scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)

NODE_CLASS_MAPPINGS = {
    "UpscaleModelLoader": UpscaleModelLoader,
    "ImageUpscaleWithModel": ImageUpscaleWithModel
}
