import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_107 = vaeloader.load_vae(vae_name="ae.safetensors")

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_112 = unetloader.load_unet(
            unet_name="flux1-fill-dev-fp8.safetensors", weight_dtype="default"
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_120 = ksamplerselect.get_sampler(sampler_name="dpmpp_2m")

        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_151 = clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )

        stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
        stylemodelloader_152 = stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_187 = loadimage.load_image(image="pexels-pixabay-68201 (2).jpg")

        cr_upscale_image = NODE_CLASS_MAPPINGS["CR Upscale Image"]()
        cr_upscale_image_190 = cr_upscale_image.upscale(
            upscale_model="4x_NMKD-Siax_200k.pth",
            resampling_method="lanczos",
            supersample="true",
            image=get_value_at_index(loadimage_187, 0),
        )

        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
        clipvisionencode_153 = clipvisionencode.encode(
            crop="none",
            clip_vision=get_value_at_index(clipvisionloader_151, 0),
            image=get_value_at_index(cr_upscale_image_190, 0),
        )

        loadconditioningnode = NODE_CLASS_MAPPINGS["LoadConditioningNode"]()
        loadconditioningnode_327 = loadconditioningnode.load_conditioning(
            filename="prompt_conditioning.safetensors"
        )

        stylemodelapplysimple = NODE_CLASS_MAPPINGS["StyleModelApplySimple"]()
        stylemodelapplysimple_106 = stylemodelapplysimple.apply_stylemodel(
            image_strength="high",
            conditioning=get_value_at_index(loadconditioningnode_327, 0),
            style_model=get_value_at_index(stylemodelloader_152, 0),
            clip_vision_output=get_value_at_index(clipvisionencode_153, 0),
        )

        conditioningconcat = NODE_CLASS_MAPPINGS["ConditioningConcat"]()
        conditioningconcat_121 = conditioningconcat.concat(
            conditioning_to=get_value_at_index(loadconditioningnode_327, 0),
            conditioning_from=get_value_at_index(stylemodelapplysimple_106, 0),
        )

        loadimage_200 = loadimage.load_image(
            image="clipspace/clipspace-mask-518788.3000000119.png [input]"
        )

        cr_upscale_image_336 = cr_upscale_image.upscale(
            upscale_model="4x_NMKD-Siax_200k.pth",
            resampling_method="lanczos",
            supersample="true",
            image=get_value_at_index(loadimage_200, 0),
        )

        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        masktoimage_170 = masktoimage.mask_to_image(
            mask=get_value_at_index(loadimage_200, 1)
        )

        cr_upscale_image_337 = cr_upscale_image.upscale(
            upscale_model="4x_NMKD-Siax_200k.pth",
            resampling_method="lanczos",
            supersample="true",
            image=get_value_at_index(masktoimage_170, 0),
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_171 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(cr_upscale_image_337, 0)
        )

        inpaintcropimproved = NODE_CLASS_MAPPINGS["InpaintCropImproved"]()
        inpaintcropimproved_183 = inpaintcropimproved.inpaint_crop(
            downscale_algorithm="bilinear",
            upscale_algorithm="bicubic",
            preresize=False,
            preresize_mode="ensure minimum resolution",
            preresize_min_width=1024,
            preresize_min_height=1024,
            preresize_max_width=16384,
            preresize_max_height=16384,
            mask_fill_holes=True,
            mask_expand_pixels=0,
            mask_invert=False,
            mask_blend_pixels=32,
            mask_hipass_filter=0.1,
            extend_for_outpainting=False,
            extend_up_factor=1,
            extend_down_factor=1,
            extend_left_factor=1,
            extend_right_factor=1,
            context_from_mask_extend_factor=1.2000000000000002,
            output_resize_to_target_size=True,
            output_target_width=1080,
            output_target_height=1080,
            output_padding="128",
            image=get_value_at_index(cr_upscale_image_336, 0),
            mask=get_value_at_index(imagetomask_171, 0),
        )

        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize+"]()
        getimagesize_179 = getimagesize.execute(
            image=get_value_at_index(inpaintcropimproved_183, 1)
        )

        imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
        imageresize_181 = imageresize.execute(
            width=16384,
            height=get_value_at_index(getimagesize_179, 1),
            interpolation="lanczos",
            method="keep proportion",
            condition="always",
            multiple_of=0,
            image=get_value_at_index(cr_upscale_image_190, 0),
        )

        imageconcanate = NODE_CLASS_MAPPINGS["ImageConcanate"]()
        imageconcanate_123 = imageconcanate.concatenate(
            direction="right",
            match_image_size=False,
            image1=get_value_at_index(imageresize_181, 0),
            image2=get_value_at_index(inpaintcropimproved_183, 1),
        )

        getimagesize_102 = getimagesize.execute(
            image=get_value_at_index(imageresize_181, 0)
        )

        cr_color_panel = NODE_CLASS_MAPPINGS["CR Color Panel"]()
        cr_color_panel_125 = cr_color_panel.make_panel(
            panel_width=get_value_at_index(getimagesize_102, 0),
            panel_height=get_value_at_index(getimagesize_102, 1),
            fill_color="black",
            fill_color_hex="#000000",
        )

        masktoimage_110 = masktoimage.mask_to_image(
            mask=get_value_at_index(inpaintcropimproved_183, 2)
        )

        imageconcanate_103 = imageconcanate.concatenate(
            direction="right",
            match_image_size=False,
            image1=get_value_at_index(cr_color_panel_125, 0),
            image2=get_value_at_index(masktoimage_110, 0),
        )

        imagetomask_115 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(imageconcanate_103, 0)
        )

        growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
        growmaskwithblur_127 = growmaskwithblur.expand_mask(
            expand=8,
            incremental_expandrate=0,
            tapered_corners=False,
            flip_input=False,
            blur_radius=8,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=False,
            mask=get_value_at_index(imagetomask_115, 0),
        )

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_160 = inpaintmodelconditioning.encode(
            noise_mask=False,
            positive=get_value_at_index(conditioningconcat_121, 0),
            negative=get_value_at_index(conditioningconcat_121, 0),
            vae=get_value_at_index(vaeloader_107, 0),
            pixels=get_value_at_index(imageconcanate_123, 0),
            mask=get_value_at_index(growmaskwithblur_127, 0),
        )

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_163 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        randomnoise_225 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        ksamplerselect_229 = ksamplerselect.get_sampler(sampler_name="dpmpp_2m")

        cr_upscale_image_318 = cr_upscale_image.upscale(
            upscale_model="4x_NMKD-Siax_200k.pth",
            resampling_method="lanczos",
            supersample="true",
            image=get_value_at_index(loadimage_187, 0),
        )

        watchdetector = NODE_CLASS_MAPPINGS["WatchDetector"]()
        watchdetector_319 = watchdetector.detect_watch(
            dp=1.2,
            param1=100,
            param2=80,
            min_dist_factor=0.2,
            min_radius_factor=0.1,
            max_radius_factor=0.4000000000000001,
            bg_red=220,
            bg_green=220,
            bg_blue=220,
            image=get_value_at_index(cr_upscale_image_318, 0),
        )

        clipvisionencode_272 = clipvisionencode.encode(
            crop="none",
            clip_vision=get_value_at_index(clipvisionloader_151, 0),
            image=get_value_at_index(watchdetector_319, 1),
        )

        stylemodelapplysimple_208 = stylemodelapplysimple.apply_stylemodel(
            image_strength="high",
            conditioning=get_value_at_index(loadconditioningnode_327, 0),
            style_model=get_value_at_index(stylemodelloader_152, 0),
            clip_vision_output=get_value_at_index(clipvisionencode_272, 0),
        )

        conditioningconcat_230 = conditioningconcat.concat(
            conditioning_to=get_value_at_index(loadconditioningnode_327, 0),
            conditioning_from=get_value_at_index(stylemodelapplysimple_208, 0),
        )

        getimagesize_134 = getimagesize.execute(
            image=get_value_at_index(inpaintcropimproved_183, 1)
        )

        getimagesize_105 = getimagesize.execute(
            image=get_value_at_index(imageconcanate_123, 0)
        )

        teacache = NODE_CLASS_MAPPINGS["TeaCache"]()
        teacache_156 = teacache.apply_teacache(
            model_type="flux",
            rel_l1_thresh=0.4,
            start_percent=0,
            end_percent=1,
            cache_device="cuda",
            model=get_value_at_index(unetloader_112, 0),
        )

        differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        differentialdiffusion_128 = differentialdiffusion.apply(
            model=get_value_at_index(teacache_156, 0)
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_344 = loraloadermodelonly.load_lora_model_only(
            lora_name="comfyui_portrait_lora64.safetensors",
            strength_model=0.8000000000000002,
            model=get_value_at_index(differentialdiffusion_128, 0),
        )

        loraloadermodelonly_345 = loraloadermodelonly.load_lora_model_only(
            lora_name="pytorch_lora_weights.safetensors",
            strength_model=0.6000000000000001,
            model=get_value_at_index(loraloadermodelonly_344, 0),
        )

        modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
        modelsamplingflux_129 = modelsamplingflux.patch(
            max_shift=1.15,
            base_shift=0.5,
            width=get_value_at_index(getimagesize_105, 0),
            height=get_value_at_index(getimagesize_105, 1),
            model=get_value_at_index(loraloadermodelonly_345, 0),
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_169 = fluxguidance.append(
            guidance=30,
            conditioning=get_value_at_index(inpaintmodelconditioning_160, 0),
        )

        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicguider_117 = basicguider.get_guider(
            model=get_value_at_index(modelsamplingflux_129, 0),
            conditioning=get_value_at_index(fluxguidance_169, 0),
        )

        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        basicscheduler_176 = basicscheduler.get_sigmas(
            scheduler="sgm_uniform",
            steps=30,
            denoise=1,
            model=get_value_at_index(modelsamplingflux_129, 0),
        )

        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        samplercustomadvanced_141 = samplercustomadvanced.sample(
            noise=get_value_at_index(randomnoise_163, 0),
            guider=get_value_at_index(basicguider_117, 0),
            sampler=get_value_at_index(ksamplerselect_120, 0),
            sigmas=get_value_at_index(basicscheduler_176, 0),
            latent_image=get_value_at_index(inpaintmodelconditioning_160, 2),
        )

        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vaedecode_157 = vaedecode.decode(
            samples=get_value_at_index(samplercustomadvanced_141, 0),
            vae=get_value_at_index(vaeloader_107, 0),
        )

        imagecrop = NODE_CLASS_MAPPINGS["ImageCrop+"]()
        imagecrop_139 = imagecrop.execute(
            width=get_value_at_index(getimagesize_134, 0),
            height=get_value_at_index(getimagesize_134, 1),
            position="right-center",
            x_offset=0,
            y_offset=0,
            image=get_value_at_index(vaedecode_157, 0),
        )

        layercolor_brightnesscontrastv2 = NODE_CLASS_MAPPINGS[
            "LayerColor: BrightnessContrastV2"
        ]()
        layercolor_brightnesscontrastv2_137 = (
            layercolor_brightnesscontrastv2.color_correct_brightness_contrast_v2(
                brightness=1.05,
                contrast=0.98,
                saturation=1.05,
                image=get_value_at_index(imagecrop_139, 0),
            )
        )

        masktoimage_135 = masktoimage.mask_to_image(
            mask=get_value_at_index(growmaskwithblur_127, 0)
        )

        imagecrop_136 = imagecrop.execute(
            width=get_value_at_index(getimagesize_134, 0),
            height=get_value_at_index(getimagesize_134, 1),
            position="right-center",
            x_offset=0,
            y_offset=0,
            image=get_value_at_index(masktoimage_135, 0),
        )

        imagetomask_138 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(imagecrop_136, 0)
        )

        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        imagecompositemasked_191 = imagecompositemasked.composite(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(inpaintcropimproved_183, 1),
            source=get_value_at_index(layercolor_brightnesscontrastv2_137, 0),
            mask=get_value_at_index(imagetomask_138, 0),
        )

        inpaintstitchimproved = NODE_CLASS_MAPPINGS["InpaintStitchImproved"]()
        inpaintstitchimproved_178 = inpaintstitchimproved.inpaint_stitch(
            stitcher=get_value_at_index(inpaintcropimproved_183, 0),
            inpainted_image=get_value_at_index(imagecompositemasked_191, 0),
        )

        cr_upscale_image_186 = cr_upscale_image.upscale(
            upscale_model="4x_NMKD-Siax_200k.pth",
            resampling_method="lanczos",
            supersample="true",
            image=get_value_at_index(inpaintstitchimproved_178, 0),
        )

        watchdetector_312 = watchdetector.detect_watch(
            dp=1.2,
            param1=100,
            param2=80,
            min_dist_factor=0.2,
            min_radius_factor=0.010000000000000002,
            max_radius_factor=0.4,
            bg_red=220,
            bg_green=220,
            bg_blue=220,
            image=get_value_at_index(cr_upscale_image_186, 0),
        )

        inpaintcropimproved_258 = inpaintcropimproved.inpaint_crop(
            downscale_algorithm="bilinear",
            upscale_algorithm="bicubic",
            preresize=False,
            preresize_mode="ensure minimum resolution",
            preresize_min_width=1024,
            preresize_min_height=1024,
            preresize_max_width=16384,
            preresize_max_height=16384,
            mask_fill_holes=True,
            mask_expand_pixels=0,
            mask_invert=False,
            mask_blend_pixels=32,
            mask_hipass_filter=0.1,
            extend_for_outpainting=False,
            extend_up_factor=1,
            extend_down_factor=1,
            extend_left_factor=1,
            extend_right_factor=1,
            context_from_mask_extend_factor=1.2000000000000002,
            output_resize_to_target_size=True,
            output_target_width=1080,
            output_target_height=1080,
            output_padding="128",
            image=get_value_at_index(cr_upscale_image_186, 0),
            mask=get_value_at_index(watchdetector_312, 0),
        )

        getimagesize_212 = getimagesize.execute(
            image=get_value_at_index(inpaintcropimproved_258, 1)
        )

        imageresize_213 = imageresize.execute(
            width=16384,
            height=get_value_at_index(getimagesize_212, 1),
            interpolation="lanczos",
            method="keep proportion",
            condition="always",
            multiple_of=0,
            image=get_value_at_index(watchdetector_319, 1),
        )

        imageconcanate_232 = imageconcanate.concatenate(
            direction="right",
            match_image_size=False,
            image1=get_value_at_index(imageresize_213, 0),
            image2=get_value_at_index(inpaintcropimproved_258, 1),
        )

        getimagesize_204 = getimagesize.execute(
            image=get_value_at_index(imageresize_213, 0)
        )

        cr_color_panel_234 = cr_color_panel.make_panel(
            panel_width=get_value_at_index(getimagesize_204, 0),
            panel_height=get_value_at_index(getimagesize_204, 1),
            fill_color="black",
            fill_color_hex="#000000",
        )

        masktoimage_211 = masktoimage.mask_to_image(
            mask=get_value_at_index(inpaintcropimproved_258, 2)
        )

        imageconcanate_205 = imageconcanate.concatenate(
            direction="right",
            match_image_size=False,
            image1=get_value_at_index(cr_color_panel_234, 0),
            image2=get_value_at_index(masktoimage_211, 0),
        )

        imagetomask_221 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(imageconcanate_205, 0)
        )

        growmaskwithblur_236 = growmaskwithblur.expand_mask(
            expand=8,
            incremental_expandrate=0,
            tapered_corners=False,
            flip_input=False,
            blur_radius=8,
            lerp_alpha=1,
            decay_factor=1,
            fill_holes=False,
            mask=get_value_at_index(imagetomask_221, 0),
        )

        inpaintmodelconditioning_239 = inpaintmodelconditioning.encode(
            noise_mask=False,
            positive=get_value_at_index(conditioningconcat_230, 0),
            negative=get_value_at_index(conditioningconcat_230, 0),
            vae=get_value_at_index(vaeloader_107, 0),
            pixels=get_value_at_index(imageconcanate_232, 0),
            mask=get_value_at_index(growmaskwithblur_236, 0),
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_322 = upscalemodelloader.load_model(
            model_name="4x_NMKD-Siax_200k.pth"
        )

        upscalemodelloader_333 = upscalemodelloader.load_model(
            model_name="4x_NMKD-Siax_200k.pth"
        )

        maskpreview = NODE_CLASS_MAPPINGS["MaskPreview+"]()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()

        for q in range(1):
            getimagesize_207 = getimagesize.execute(
                image=get_value_at_index(imageconcanate_232, 0)
            )

            fluxguidance_223 = fluxguidance.append(
                guidance=50,
                conditioning=get_value_at_index(inpaintmodelconditioning_239, 0),
            )

            modelsamplingflux_238 = modelsamplingflux.patch(
                max_shift=1.15,
                base_shift=0.5,
                width=get_value_at_index(getimagesize_207, 0),
                height=get_value_at_index(getimagesize_207, 1),
                model=get_value_at_index(loraloadermodelonly_345, 0),
            )

            basicguider_224 = basicguider.get_guider(
                model=get_value_at_index(modelsamplingflux_238, 0),
                conditioning=get_value_at_index(fluxguidance_223, 0),
            )

            getimagesize_245 = getimagesize.execute(
                image=get_value_at_index(inpaintcropimproved_258, 1)
            )

            masktoimage_246 = masktoimage.mask_to_image(
                mask=get_value_at_index(growmaskwithblur_236, 0)
            )

            imagecrop_247 = imagecrop.execute(
                width=get_value_at_index(getimagesize_245, 0),
                height=get_value_at_index(getimagesize_245, 1),
                position="right-center",
                x_offset=0,
                y_offset=0,
                image=get_value_at_index(masktoimage_246, 0),
            )

            imagetomask_249 = imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(imagecrop_247, 0)
            )

            basicscheduler_270 = basicscheduler.get_sigmas(
                scheduler="sgm_uniform",
                steps=30,
                denoise=1,
                model=get_value_at_index(modelsamplingflux_238, 0),
            )

            samplercustomadvanced_253 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_225, 0),
                guider=get_value_at_index(basicguider_224, 0),
                sampler=get_value_at_index(ksamplerselect_229, 0),
                sigmas=get_value_at_index(basicscheduler_270, 0),
                latent_image=get_value_at_index(inpaintmodelconditioning_239, 2),
            )

            vaedecode_278 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_253, 0),
                vae=get_value_at_index(vaeloader_107, 0),
            )

            imagecrop_251 = imagecrop.execute(
                width=get_value_at_index(getimagesize_245, 0),
                height=get_value_at_index(getimagesize_245, 1),
                position="right-center",
                x_offset=0,
                y_offset=0,
                image=get_value_at_index(vaedecode_278, 0),
            )

            imagecompositemasked_273 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(inpaintcropimproved_258, 1),
                source=get_value_at_index(imagecrop_251, 0),
                mask=get_value_at_index(imagetomask_249, 0),
            )

            inpaintstitchimproved_262 = inpaintstitchimproved.inpaint_stitch(
                stitcher=get_value_at_index(inpaintcropimproved_258, 0),
                inpainted_image=get_value_at_index(imagecompositemasked_273, 0),
            )

            maskpreview_287 = maskpreview.execute(
                mask=get_value_at_index(watchdetector_312, 0)
            )

            imageupscalewithmodel_324 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_322, 0),
                image=get_value_at_index(inpaintstitchimproved_262, 0),
            )

            maskpreview_317 = maskpreview.execute(
                mask=get_value_at_index(watchdetector_319, 0)
            )

            imageupscalewithmodel_334 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_333, 0),
                image=get_value_at_index(inpaintstitchimproved_178, 0),
            )


if __name__ == "__main__":
    main()
