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
        # --- Start of Modification ---
        # Define the maximum dimension (height or width) for the upscaled image.
        MAX_DIMENSION = 2160

        h, w = image.shape[1:3]
        final_scale = 1.0 # Default to 1.0 (no upscale)

        # Proceed only if the original image has dimensions.
        if h > 0 and w > 0:
            # If the original image's largest dimension already exceeds the limit, set scale to 1.0.
            if max(h, w) >= MAX_DIMENSION:
                final_scale = 1.0
            else:
                # Calculate the maximum scale factor that can be applied without exceeding MAX_DIMENSION.
                scale_limit = MAX_DIMENSION / max(h, w)
                # Use the smaller value between the model's native scale and our calculated limit.
                final_scale = min(upscale_model.scale, scale_limit)
        # --- End of Modification ---

        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        # MODIFIED: Use the calculated 'final_scale' for memory estimation.
        memory_required += (512 * 512 * 3) * image.element_size() * max(final_scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        tile = min(h, w, 512)
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                # MODIFIED: Use the calculated 'final_scale' for the upscaling operation.
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
