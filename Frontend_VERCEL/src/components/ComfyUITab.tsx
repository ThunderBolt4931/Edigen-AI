import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { toast } from 'sonner';
import { BACKEND_COMFYUI_URL } from '@/config/backend_comfyui';

type WorkflowJSON1 = Record<string, any>;
type WorkflowJSON2 = Record<string, any>;
const WORKFLOW_JSON_1: WorkflowJSON1 = {
  "102": {
    "inputs": {
      "image": [
        "181",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "103": {
    "inputs": {
      "direction": "right",
      "match_image_size": false,
      "image1": [
        "125",
        0
      ],
      "image2": [
        "110",
        0
      ]
    },
    "class_type": "ImageConcanate",
    "_meta": {
      "title": "Image Concatenate"
    }
  },
  "105": {
    "inputs": {
      "image": [
        "123",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "106": {
    "inputs": {
      "image_strength": "high",
      "conditioning": [
        "327",
        0
      ],
      "style_model": [
        "152",
        0
      ],
      "clip_vision_output": [
        "153",
        0
      ]
    },
    "class_type": "StyleModelApplySimple",
    "_meta": {
      "title": "StyleModelApplySimple"
    }
  },
  "107": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "110": {
    "inputs": {
      "mask": [
        "183",
        2
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "112": {
    "inputs": {
      "unet_name": "flux1-fill-dev-fp8.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "115": {
    "inputs": {
      "channel": "red",
      "image": [
        "103",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "116": {
    "inputs": {
      "images": [
        "103",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "117": {
    "inputs": {
      "model": [
        "129",
        0
      ],
      "conditioning": [
        "169",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "120": {
    "inputs": {
      "sampler_name": "dpmpp_2m"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "121": {
    "inputs": {
      "conditioning_to": [
        "327",
        0
      ],
      "conditioning_from": [
        "106",
        0
      ]
    },
    "class_type": "ConditioningConcat",
    "_meta": {
      "title": "Conditioning (Concat)"
    }
  },
  "123": {
    "inputs": {
      "direction": "right",
      "match_image_size": false,
      "image1": [
        "181",
        0
      ],
      "image2": [
        "183",
        1
      ]
    },
    "class_type": "ImageConcanate",
    "_meta": {
      "title": "Image Concatenate"
    }
  },
  "125": {
    "inputs": {
      "panel_width": [
        "102",
        0
      ],
      "panel_height": [
        "102",
        1
      ],
      "fill_color": "black",
      "fill_color_hex": "#000000"
    },
    "class_type": "CR Color Panel",
    "_meta": {
      "title": "🌁 CR Color Panel"
    }
  },
  "127": {
    "inputs": {
      "expand": 8,
      "incremental_expandrate": 0,
      "tapered_corners": false,
      "flip_input": false,
      "blur_radius": 8,
      "lerp_alpha": 1,
      "decay_factor": 1,
      "fill_holes": false,
      "mask": [
        "115",
        0
      ]
    },
    "class_type": "GrowMaskWithBlur",
    "_meta": {
      "title": "Grow Mask With Blur"
    }
  },
  "128": {
    "inputs": {
      "model": [
        "156",
        0
      ]
    },
    "class_type": "DifferentialDiffusion",
    "_meta": {
      "title": "Differential Diffusion"
    }
  },
  "129": {
    "inputs": {
      "max_shift": 1.15,
      "base_shift": 0.5,
      "width": [
        "105",
        0
      ],
      "height": [
        "105",
        1
      ],
      "model": [
        "344",
        0
      ]
    },
    "class_type": "ModelSamplingFlux",
    "_meta": {
      "title": "ModelSamplingFlux"
    }
  },
  "134": {
    "inputs": {
      "image": [
        "183",
        1
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "135": {
    "inputs": {
      "mask": [
        "127",
        0
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "136": {
    "inputs": {
      "width": [
        "134",
        0
      ],
      "height": [
        "134",
        1
      ],
      "position": "right-center",
      "x_offset": 0,
      "y_offset": 0,
      "image": [
        "135",
        0
      ]
    },
    "class_type": "ImageCrop+",
    "_meta": {
      "title": "🔧 Image Crop"
    }
  },
  "137": {
    "inputs": {
      "brightness": 1.05,
      "contrast": 0.98,
      "saturation": 1.05,
      "image": [
        "139",
        0
      ]
    },
    "class_type": "LayerColor: BrightnessContrastV2",
    "_meta": {
      "title": "LayerColor: Brightness Contrast V2"
    }
  },
  "138": {
    "inputs": {
      "channel": "red",
      "image": [
        "136",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "139": {
    "inputs": {
      "width": [
        "134",
        0
      ],
      "height": [
        "134",
        1
      ],
      "position": "right-center",
      "x_offset": 0,
      "y_offset": 0,
      "image": [
        "157",
        0
      ]
    },
    "class_type": "ImageCrop+",
    "_meta": {
      "title": "🔧 Image Crop"
    }
  },
  "141": {
    "inputs": {
      "noise": [
        "163",
        0
      ],
      "guider": [
        "117",
        0
      ],
      "sampler": [
        "120",
        0
      ],
      "sigmas": [
        "176",
        0
      ],
      "latent_image": [
        "160",
        2
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "151": {
    "inputs": {
      "clip_name": "sigclip_vision_patch14_384.safetensors"
    },
    "class_type": "CLIPVisionLoader",
    "_meta": {
      "title": "Load CLIP Vision"
    }
  },
  "152": {
    "inputs": {
      "style_model_name": "flux1-redux-dev.safetensors"
    },
    "class_type": "StyleModelLoader",
    "_meta": {
      "title": "Load Style Model"
    }
  },
  "153": {
    "inputs": {
      "crop": "none",
      "clip_vision": [
        "151",
        0
      ],
      "image": [
        "190",
        0
      ]
    },
    "class_type": "CLIPVisionEncode",
    "_meta": {
      "title": "CLIP Vision Encode"
    }
  },
  "156": {
    "inputs": {
      "model_type": "flux",
      "rel_l1_thresh": 0.4,
      "start_percent": 0,
      "end_percent": 1,
      "cache_device": "cuda",
      "model": [
        "112",
        0
      ]
    },
    "class_type": "TeaCache",
    "_meta": {
      "title": "TeaCache"
    }
  },
  "157": {
    "inputs": {
      "samples": [
        "141",
        0
      ],
      "vae": [
        "107",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "160": {
    "inputs": {
      "noise_mask": false,
      "positive": [
        "121",
        0
      ],
      "negative": [
        "121",
        0
      ],
      "vae": [
        "107",
        0
      ],
      "pixels": [
        "123",
        0
      ],
      "mask": [
        "127",
        0
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "163": {
    "inputs": {
      "noise_seed": 993192807042492
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "169": {
    "inputs": {
      "guidance": 30,
      "conditioning": [
        "160",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "176": {
    "inputs": {
      "scheduler": "sgm_uniform",
      "steps": 15,
      "denoise": 1,
      "model": [
        "129",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "178": {
    "inputs": {
      "stitcher": [
        "183",
        0
      ],
      "inpainted_image": [
        "191",
        0
      ]
    },
    "class_type": "InpaintStitchImproved",
    "_meta": {
      "title": "✂️ Inpaint Stitch (Improved)"
    }
  },
  "179": {
    "inputs": {
      "image": [
        "183",
        1
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "181": {
    "inputs": {
      "width": 16384,
      "height": [
        "179",
        1
      ],
      "interpolation": "lanczos",
      "method": "keep proportion",
      "condition": "always",
      "multiple_of": 0,
      "image": [
        "190",
        0
      ]
    },
    "class_type": "ImageResize+",
    "_meta": {
      "title": "🔧 Image Resize"
    }
  },
  "183": {
    "inputs": {
      "downscale_algorithm": "bilinear",
      "upscale_algorithm": "bicubic",
      "preresize": false,
      "preresize_mode": "ensure minimum resolution",
      "preresize_min_width": 1024,
      "preresize_min_height": 1024,
      "preresize_max_width": 16384,
      "preresize_max_height": 16384,
      "mask_fill_holes": true,
      "mask_expand_pixels": 0,
      "mask_invert": false,
      "mask_blend_pixels": 32,
      "mask_hipass_filter": 0.1,
      "extend_for_outpainting": false,
      "extend_up_factor": 1,
      "extend_down_factor": 1,
      "extend_left_factor": 1,
      "extend_right_factor": 1,
      "context_from_mask_extend_factor": 1.2000000000000002,
      "output_resize_to_target_size": true,
      "output_target_width": 512,
      "output_target_height": 512,
      "output_padding": "128",
      "image": [
        "200",
        0
      ],
      "mask": [
        "358",
        0
      ]
    },
    "class_type": "InpaintCropImproved",
    "_meta": {
      "title": "✂️ Inpaint Crop (Improved)"
    }
  },
  "186": {
    "inputs": {
      "upscale_model": "4x_NMKD-Siax_200k.pth",
      "resampling_method": "lanczos",
      "supersample": "true",
      "image": [
        "178",
        0
      ]
    },
    "class_type": "CR Upscale Image",
    "_meta": {
      "title": "🔍 CR Upscale Image"
    }
  },
  "187": {
    "inputs": {
      "image": "watch_6.jpg"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Insert object"
    }
  },
  "190": {
    "inputs": {
      "upscale_model": "4x_NMKD-Siax_200k.pth",
      "resampling_method": "lanczos",
      "supersample": "true",
      "image": [
        "187",
        0
      ]
    },
    "class_type": "CR Upscale Image",
    "_meta": {
      "title": "🔍 CR Upscale Image"
    }
  },
  "191": {
    "inputs": {
      "x": 0,
      "y": 0,
      "resize_source": false,
      "destination": [
        "183",
        1
      ],
      "source": [
        "137",
        0
      ],
      "mask": [
        "138",
        0
      ]
    },
    "class_type": "ImageCompositeMasked",
    "_meta": {
      "title": "ImageCompositeMasked"
    }
  },
  "192": {
    "inputs": {
      "images": [
        "191",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "200": {
    "inputs": {
      "image": "clipspace/clipspace-mask-15698059.599999964.png [input]"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Insert hand"
    }
  },
  "204": {
    "inputs": {
      "image": [
        "213",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "205": {
    "inputs": {
      "direction": "right",
      "match_image_size": false,
      "image1": [
        "234",
        0
      ],
      "image2": [
        "211",
        0
      ]
    },
    "class_type": "ImageConcanate",
    "_meta": {
      "title": "Image Concatenate"
    }
  },
  "207": {
    "inputs": {
      "image": [
        "232",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "208": {
    "inputs": {
      "image_strength": "high",
      "conditioning": [
        "351",
        0
      ],
      "style_model": [
        "152",
        0
      ],
      "clip_vision_output": [
        "272",
        0
      ]
    },
    "class_type": "StyleModelApplySimple",
    "_meta": {
      "title": "StyleModelApplySimple"
    }
  },
  "211": {
    "inputs": {
      "mask": [
        "258",
        2
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "212": {
    "inputs": {
      "image": [
        "258",
        1
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "213": {
    "inputs": {
      "width": 16384,
      "height": [
        "212",
        1
      ],
      "interpolation": "lanczos",
      "method": "keep proportion",
      "condition": "always",
      "multiple_of": 0,
      "image": [
        "319",
        1
      ]
    },
    "class_type": "ImageResize+",
    "_meta": {
      "title": "🔧 Image Resize"
    }
  },
  "221": {
    "inputs": {
      "channel": "red",
      "image": [
        "205",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "222": {
    "inputs": {
      "images": [
        "205",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "223": {
    "inputs": {
      "guidance": 50,
      "conditioning": [
        "239",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "224": {
    "inputs": {
      "model": [
        "238",
        0
      ],
      "conditioning": [
        "223",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "225": {
    "inputs": {
      "noise_seed": 883063622390318
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "229": {
    "inputs": {
      "sampler_name": "dpmpp_2m"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "230": {
    "inputs": {
      "conditioning_to": [
        "327",
        0
      ],
      "conditioning_from": [
        "208",
        0
      ]
    },
    "class_type": "ConditioningConcat",
    "_meta": {
      "title": "Conditioning (Concat)"
    }
  },
  "232": {
    "inputs": {
      "direction": "right",
      "match_image_size": false,
      "image1": [
        "213",
        0
      ],
      "image2": [
        "258",
        1
      ]
    },
    "class_type": "ImageConcanate",
    "_meta": {
      "title": "Image Concatenate"
    }
  },
  "234": {
    "inputs": {
      "panel_width": [
        "204",
        0
      ],
      "panel_height": [
        "204",
        1
      ],
      "fill_color": "black",
      "fill_color_hex": "#000000"
    },
    "class_type": "CR Color Panel",
    "_meta": {
      "title": "🌁 CR Color Panel"
    }
  },
  "236": {
    "inputs": {
      "expand": 8,
      "incremental_expandrate": 0,
      "tapered_corners": false,
      "flip_input": false,
      "blur_radius": 8,
      "lerp_alpha": 1,
      "decay_factor": 1,
      "fill_holes": false,
      "mask": [
        "221",
        0
      ]
    },
    "class_type": "GrowMaskWithBlur",
    "_meta": {
      "title": "Grow Mask With Blur"
    }
  },
  "238": {
    "inputs": {
      "max_shift": 1.15,
      "base_shift": 0.5,
      "width": [
        "207",
        0
      ],
      "height": [
        "207",
        1
      ],
      "model": [
        "344",
        0
      ]
    },
    "class_type": "ModelSamplingFlux",
    "_meta": {
      "title": "ModelSamplingFlux"
    }
  },
  "239": {
    "inputs": {
      "noise_mask": false,
      "positive": [
        "230",
        0
      ],
      "negative": [
        "230",
        0
      ],
      "vae": [
        "107",
        0
      ],
      "pixels": [
        "232",
        0
      ],
      "mask": [
        "236",
        0
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "245": {
    "inputs": {
      "image": [
        "258",
        1
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "246": {
    "inputs": {
      "mask": [
        "236",
        0
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "247": {
    "inputs": {
      "width": [
        "245",
        0
      ],
      "height": [
        "245",
        1
      ],
      "position": "right-center",
      "x_offset": 0,
      "y_offset": 0,
      "image": [
        "246",
        0
      ]
    },
    "class_type": "ImageCrop+",
    "_meta": {
      "title": "🔧 Image Crop"
    }
  },
  "249": {
    "inputs": {
      "channel": "red",
      "image": [
        "247",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "251": {
    "inputs": {
      "width": [
        "245",
        0
      ],
      "height": [
        "245",
        1
      ],
      "position": "right-center",
      "x_offset": 0,
      "y_offset": 0,
      "image": [
        "278",
        0
      ]
    },
    "class_type": "ImageCrop+",
    "_meta": {
      "title": "🔧 Image Crop"
    }
  },
  "253": {
    "inputs": {
      "noise": [
        "225",
        0
      ],
      "guider": [
        "224",
        0
      ],
      "sampler": [
        "229",
        0
      ],
      "sigmas": [
        "270",
        0
      ],
      "latent_image": [
        "239",
        2
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "258": {
    "inputs": {
      "downscale_algorithm": "bilinear",
      "upscale_algorithm": "bicubic",
      "preresize": false,
      "preresize_mode": "ensure minimum resolution",
      "preresize_min_width": 1024,
      "preresize_min_height": 1024,
      "preresize_max_width": 16384,
      "preresize_max_height": 16384,
      "mask_fill_holes": true,
      "mask_expand_pixels": 0,
      "mask_invert": false,
      "mask_blend_pixels": 32,
      "mask_hipass_filter": 0.1,
      "extend_for_outpainting": false,
      "extend_up_factor": 1,
      "extend_down_factor": 1,
      "extend_left_factor": 1,
      "extend_right_factor": 1,
      "context_from_mask_extend_factor": 1.2000000000000002,
      "output_resize_to_target_size": true,
      "output_target_width": 720,
      "output_target_height": 720,
      "output_padding": "128",
      "image": [
        "186",
        0
      ],
      "mask": [
        "356",
        0
      ]
    },
    "class_type": "InpaintCropImproved",
    "_meta": {
      "title": "✂️ Inpaint Crop (Improved)"
    }
  },
  "262": {
    "inputs": {
      "stitcher": [
        "258",
        0
      ],
      "inpainted_image": [
        "273",
        0
      ]
    },
    "class_type": "InpaintStitchImproved",
    "_meta": {
      "title": "✂️ Inpaint Stitch (Improved)"
    }
  },
  "270": {
    "inputs": {
      "scheduler": "sgm_uniform",
      "steps": 30,
      "denoise": 1,
      "model": [
        "238",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "272": {
    "inputs": {
      "crop": "none",
      "clip_vision": [
        "151",
        0
      ],
      "image": [
        "319",
        1
      ]
    },
    "class_type": "CLIPVisionEncode",
    "_meta": {
      "title": "CLIP Vision Encode"
    }
  },
  "273": {
    "inputs": {
      "x": 0,
      "y": 0,
      "resize_source": false,
      "destination": [
        "258",
        1
      ],
      "source": [
        "251",
        0
      ],
      "mask": [
        "249",
        0
      ]
    },
    "class_type": "ImageCompositeMasked",
    "_meta": {
      "title": "ImageCompositeMasked"
    }
  },
  "278": {
    "inputs": {
      "samples": [
        "253",
        0
      ],
      "vae": [
        "107",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "282": {
    "inputs": {
      "images": [
        "232",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "287": {
    "inputs": {
      "mask": [
        "356",
        0
      ]
    },
    "class_type": "MaskPreview+",
    "_meta": {
      "title": "🔧 Mask Preview"
    }
  },
  "289": {
    "inputs": {
      "images": [
        "319",
        1
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "308": {
    "inputs": {
      "images": [
        "324",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Final Output"
    }
  },
  "309": {
    "inputs": {
      "images": [
        "123",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "316": {
    "inputs": {},
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "317": {
    "inputs": {
      "mask": [
        "319",
        0
      ]
    },
    "class_type": "MaskPreview+",
    "_meta": {
      "title": "🔧 Mask Preview"
    }
  },
  "319": {
    "inputs": {
      "dp": 1.2,
      "param1": 100,
      "param2": 80,
      "min_dist_factor": 0.2,
      "min_radius_factor": 0.1,
      "max_radius_factor": 0.4000000000000001,
      "bg_red": 220,
      "bg_green": 220,
      "bg_blue": 220,
      "image": [
        "190",
        0
      ]
    },
    "class_type": "WatchDetector",
    "_meta": {
      "title": "Watch Detector"
    }
  },
  "322": {
    "inputs": {
      "model_name": "4x_NMKD-Siax_200k.pth"
    },
    "class_type": "UpscaleModelLoader",
    "_meta": {
      "title": "Load Upscale Model"
    }
  },
  "324": {
    "inputs": {
      "upscale_model": [
        "322",
        0
      ],
      "image": [
        "262",
        0
      ]
    },
    "class_type": "ImageUpscaleWithModel",
    "_meta": {
      "title": "Upscale Image (using Model)"
    }
  },
  "327": {
    "inputs": {
      "filename": "prompt_conditioning.safetensors"
    },
    "class_type": "LoadConditioningNode",
    "_meta": {
      "title": "Load Conditioning"
    }
  },
  "333": {
    "inputs": {
      "model_name": "4x_NMKD-Siax_200k.pth"
    },
    "class_type": "UpscaleModelLoader",
    "_meta": {
      "title": "Load Upscale Model"
    }
  },
  "334": {
    "inputs": {
      "upscale_model": [
        "333",
        0
      ],
      "image": [
        "178",
        0
      ]
    },
    "class_type": "ImageUpscaleWithModel",
    "_meta": {
      "title": "Upscale Image (using Model)"
    }
  },
  "335": {
    "inputs": {
      "images": [
        "334",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Image Pipeline 1"
    }
  },
  "344": {
    "inputs": {
      "lora_name": "comfyui_portrait_lora64.safetensors",
      "strength_model": 0.8000000000000002,
      "model": [
        "128",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "347": {
    "inputs": {
      "filename_prefix": "Image Pipeline 1",
      "images": [
        "334",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "350": {
    "inputs": {
      "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "351": {
    "inputs": {
      "text": [
        "352",
        0
      ],
      "clip": [
        "350",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "352": {
    "inputs": {
      "value": "'BALMAIN' text is above part of the watch inside and 'SWISS MADE' is below inside the watch face in clear, detailed text. The positions of the text are same as the image. The numbers and other features are highly detailed too."
    },
    "class_type": "PrimitiveString",
    "_meta": {
      "title": "String"
    }
  },
  "353": {
    "inputs": {
      "image": "1.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Insert Mask"
    }
  },
  "356": {
    "inputs": {
      "prompt": "",
      "threshold": 0.4,
      "smooth": 9,
      "dilate": 0,
      "blur": 0,
      "clip_seg": [
        "357",
        0
      ],
      "image": [
        "186",
        0
      ]
    },
    "class_type": "ApplyCLIPSeg+",
    "_meta": {
      "title": "🔧 Apply CLIPSeg"
    }
  },
  "357": {
    "inputs": {},
    "class_type": "LoadCLIPSegModels+",
    "_meta": {
      "title": "🔧 Load CLIPSeg Models"
    }
  },
  "358": {
    "inputs": {
      "channel": "red",
      "image": [
        "353",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  }
};
const WORKFLOW_JSON_2: WorkflowJSON2 = {
  "1": {
    "inputs": {
      "image": [
        "35",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "2": {
    "inputs": {
      "direction": "right",
      "match_image_size": false,
      "image1": [
        "15",
        0
      ],
      "image2": [
        "6",
        0
      ]
    },
    "class_type": "ImageConcanate",
    "_meta": {
      "title": "Image Concatenate"
    }
  },
  "3": {
    "inputs": {
      "image": [
        "41",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "4": {
    "inputs": {
      "image_strength": "high",
      "conditioning": [
        "88",
        0
      ],
      "style_model": [
        "49",
        0
      ],
      "clip_vision_output": [
        "38",
        0
      ]
    },
    "class_type": "StyleModelApplySimple",
    "_meta": {
      "title": "StyleModelApplySimple"
    }
  },
  "6": {
    "inputs": {
      "mask": [
        "74",
        2
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "7": {
    "inputs": {
      "channel": "red",
      "image": [
        "2",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "8": {
    "inputs": {
      "images": [
        "2",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "9": {
    "inputs": {
      "model": [
        "17",
        0
      ],
      "conditioning": [
        "69",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "12": {
    "inputs": {
      "sampler_name": "dpmpp_2m"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "13": {
    "inputs": {
      "conditioning_to": [
        "88",
        0
      ],
      "conditioning_from": [
        "4",
        0
      ]
    },
    "class_type": "ConditioningConcat",
    "_meta": {
      "title": "Conditioning (Concat)"
    }
  },
  "15": {
    "inputs": {
      "panel_width": [
        "1",
        0
      ],
      "panel_height": [
        "1",
        1
      ],
      "fill_color": "black",
      "fill_color_hex": "#000000"
    },
    "class_type": "CR Color Panel",
    "_meta": {
      "title": "🌁 CR Color Panel"
    }
  },
  "16": {
    "inputs": {
      "expand": 8,
      "incremental_expandrate": 0,
      "tapered_corners": false,
      "flip_input": false,
      "blur_radius": 8,
      "lerp_alpha": 0.24000000000000005,
      "decay_factor": 1,
      "fill_holes": false,
      "mask": [
        "7",
        0
      ]
    },
    "class_type": "GrowMaskWithBlur",
    "_meta": {
      "title": "Grow Mask With Blur"
    }
  },
  "17": {
    "inputs": {
      "max_shift": 1.15,
      "base_shift": 0.5,
      "width": [
        "3",
        0
      ],
      "height": [
        "3",
        1
      ],
      "model": [
        "73",
        0
      ]
    },
    "class_type": "ModelSamplingFlux",
    "_meta": {
      "title": "ModelSamplingFlux"
    }
  },
  "19": {
    "inputs": {
      "image": [
        "74",
        1
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "20": {
    "inputs": {
      "mask": [
        "16",
        0
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "21": {
    "inputs": {
      "width": [
        "19",
        0
      ],
      "height": [
        "19",
        1
      ],
      "position": "right-center",
      "x_offset": 0,
      "y_offset": 0,
      "image": [
        "20",
        0
      ]
    },
    "class_type": "ImageCrop+",
    "_meta": {
      "title": "🔧 Image Crop"
    }
  },
  "22": {
    "inputs": {
      "brightness": 1.05,
      "contrast": 0.98,
      "saturation": 1.05,
      "image": [
        "24",
        0
      ]
    },
    "class_type": "LayerColor: BrightnessContrastV2",
    "_meta": {
      "title": "LayerColor: Brightness Contrast V2"
    }
  },
  "23": {
    "inputs": {
      "channel": "red",
      "image": [
        "21",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "24": {
    "inputs": {
      "width": [
        "19",
        0
      ],
      "height": [
        "19",
        1
      ],
      "position": "right-center",
      "x_offset": 0,
      "y_offset": 0,
      "image": [
        "66",
        0
      ]
    },
    "class_type": "ImageCrop+",
    "_meta": {
      "title": "🔧 Image Crop"
    }
  },
  "26": {
    "inputs": {
      "noise": [
        "68",
        0
      ],
      "guider": [
        "9",
        0
      ],
      "sampler": [
        "12",
        0
      ],
      "sigmas": [
        "78",
        0
      ],
      "latent_image": [
        "33",
        2
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "33": {
    "inputs": {
      "noise_mask": false,
      "positive": [
        "13",
        0
      ],
      "negative": [
        "13",
        0
      ],
      "vae": [
        "51",
        0
      ],
      "pixels": [
        "41",
        0
      ],
      "mask": [
        "16",
        0
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "35": {
    "inputs": {
      "width": 16384,
      "height": [
        "40",
        1
      ],
      "interpolation": "lanczos",
      "method": "keep proportion",
      "condition": "always",
      "multiple_of": 0,
      "image": [
        "75",
        0
      ]
    },
    "class_type": "ImageResize+",
    "_meta": {
      "title": "🔧 Image Resize"
    }
  },
  "36": {
    "inputs": {
      "x": 0,
      "y": 0,
      "resize_source": false,
      "destination": [
        "74",
        1
      ],
      "source": [
        "22",
        0
      ],
      "mask": [
        "23",
        0
      ]
    },
    "class_type": "ImageCompositeMasked",
    "_meta": {
      "title": "ImageCompositeMasked"
    }
  },
  "38": {
    "inputs": {
      "crop": "none",
      "clip_vision": [
        "47",
        0
      ],
      "image": [
        "75",
        0
      ]
    },
    "class_type": "CLIPVisionEncode",
    "_meta": {
      "title": "CLIP Vision Encode"
    }
  },
  "39": {
    "inputs": {
      "unet_name": "flux1-fill-dev-fp8.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "40": {
    "inputs": {
      "image": [
        "74",
        1
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "41": {
    "inputs": {
      "direction": "right",
      "match_image_size": false,
      "image1": [
        "35",
        0
      ],
      "image2": [
        "74",
        1
      ]
    },
    "class_type": "ImageConcanate",
    "_meta": {
      "title": "Image Concatenate"
    }
  },
  "47": {
    "inputs": {
      "clip_name": "sigclip_vision_patch14_384.safetensors"
    },
    "class_type": "CLIPVisionLoader",
    "_meta": {
      "title": "Load CLIP Vision"
    }
  },
  "49": {
    "inputs": {
      "style_model_name": "flux1-redux-dev.safetensors"
    },
    "class_type": "StyleModelLoader",
    "_meta": {
      "title": "Load Style Model"
    }
  },
  "51": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "56": {
    "inputs": {
      "images": [
        "41",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "57": {
    "inputs": {
      "model_type": "flux",
      "rel_l1_thresh": 0.4,
      "start_percent": 0,
      "end_percent": 1,
      "cache_device": "cuda",
      "model": [
        "39",
        0
      ]
    },
    "class_type": "TeaCache",
    "_meta": {
      "title": "TeaCache"
    }
  },
  "58": {
    "inputs": {
      "filename": "prompt_conditioning.safetensors"
    },
    "class_type": "LoadConditioningNode",
    "_meta": {
      "title": "Load Conditioning"
    }
  },
  "61": {
    "inputs": {
      "stitcher": [
        "74",
        0
      ],
      "inpainted_image": [
        "36",
        0
      ]
    },
    "class_type": "InpaintStitchImproved",
    "_meta": {
      "title": "✂️ Inpaint Stitch (Improved)"
    }
  },
  "66": {
    "inputs": {
      "samples": [
        "26",
        0
      ],
      "vae": [
        "51",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "68": {
    "inputs": {
      "noise_seed": 1002766470811437
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "69": {
    "inputs": {
      "guidance": 30,
      "conditioning": [
        "33",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "70": {
    "inputs": {
      "images": [
        "36",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "71": {
    "inputs": {
      "model": [
        "57",
        0
      ]
    },
    "class_type": "DifferentialDiffusion",
    "_meta": {
      "title": "Differential Diffusion"
    }
  },
  "72": {
    "inputs": {
      "lora_name": "comfyui_portrait_lora64.safetensors",
      "strength_model": 0.8000000000000002,
      "model": [
        "71",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "73": {
    "inputs": {
      "lora_name": "pytorch_lora_weights.safetensors",
      "strength_model": 0.6000000000000001,
      "model": [
        "72",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "74": {
    "inputs": {
      "downscale_algorithm": "bilinear",
      "upscale_algorithm": "bicubic",
      "preresize": false,
      "preresize_mode": "ensure minimum resolution",
      "preresize_min_width": 1024,
      "preresize_min_height": 1024,
      "preresize_max_width": 16384,
      "preresize_max_height": 16384,
      "mask_fill_holes": true,
      "mask_expand_pixels": 0,
      "mask_invert": false,
      "mask_blend_pixels": 32,
      "mask_hipass_filter": 0.1,
      "extend_for_outpainting": false,
      "extend_up_factor": 1,
      "extend_down_factor": 1,
      "extend_left_factor": 1,
      "extend_right_factor": 1,
      "context_from_mask_extend_factor": 1.2000000000000002,
      "output_resize_to_target_size": true,
      "output_target_width": 720,
      "output_target_height": 720,
      "output_padding": "128",
      "image": [
        "93",
        0
      ],
      "mask": [
        "84",
        0
      ]
    },
    "class_type": "InpaintCropImproved",
    "_meta": {
      "title": "✂️ Inpaint Crop (Improved)"
    }
  },
  "75": {
    "inputs": {
      "image": "IMG_20250727_004653.jpg"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Other insert object"
    }
  },
  "76": {
    "inputs": {
      "image": "download.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Other insert"
    }
  },
  "78": {
    "inputs": {
      "scheduler": "sgm_uniform",
      "steps": 30,
      "denoise": 1,
      "model": [
        "17",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "84": {
    "inputs": {
      "channel": "red",
      "image": [
        "85",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "85": {
    "inputs": {
      "mask": [
        "91",
        0
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "87": {
    "inputs": {
      "clip_name1": "clip_l.safetensors",
      "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "88": {
    "inputs": {
      "text": "cap on head of the men",
      "clip": [
        "87",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "89": {
    "inputs": {
      "images": [
        "85",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "90": {
    "inputs": {
      "image": "1.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Other Mask"
    }
  },
  "91": {
    "inputs": {
      "channel": "red",
      "image": [
        "90",
        0
      ]
    },
    "class_type": "ImageToMask",
    "_meta": {
      "title": "Convert Image to Mask"
    }
  },
  "92": {
    "inputs": {
      "image": [
        "85",
        0
      ]
    },
    "class_type": "GetImageSize+",
    "_meta": {
      "title": "🔧 Get Image Size"
    }
  },
  "93": {
    "inputs": {
      "width": [
        "92",
        0
      ],
      "height": [
        "92",
        1
      ],
      "upscale_method": "nearest-exact",
      "keep_proportion": "stretch",
      "pad_color": "0, 0, 0",
      "crop_position": "center",
      "divisible_by": 2,
      "device": "cpu",
      "image": [
        "76",
        0
      ]
    },
    "class_type": "ImageResizeKJv2",
    "_meta": {
      "title": "Resize Image v2"
    }
  },
  "95": {
    "inputs": {
      "images": [
        "61",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Final Output"
    }
  }
};
export const ComfyUITab = () => {
  const [destinationImage, setDestinationImage] = useState<File | null>(null);
  const [zoom, setZoom] = useState(1);
  const lastPanPoint = useRef({ x: 0, y: 0 });
  const [maskImage, setMaskImage] = useState<File | null>(null);
  const [objectImage, setObjectImage] = useState<File | null>(null);
  const [selectedAccessory, setSelectedAccessory] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState('Status: Waiting for input...');
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  // Canvas editing state
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [brushType, setBrushType] = useState<'round' | 'square'>('round');
  const [brushColor, setBrushColor] = useState<'black' | 'white'>('black');
  const [brushSize, setBrushSize] = useState(10);
  const [isDrawing, setIsDrawing] = useState(false);
  const [editedImageData, setEditedImageData] = useState<string | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const contextRef = useRef<CanvasRenderingContext2D | null>(null);
  const originalImageRef = useRef<HTMLImageElement | null>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const isPanning = useRef(false);

  const updateStatus = useCallback((message: string, isError = false) => {
    console.log(message);
    setStatus(message);
    if (isError) {
      toast.error(message);
    }
  }, []);

  const uploadImage = useCallback(async (file: File) => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('overwrite', 'true');
    formData.append('type', 'input');

    updateStatus(`Uploading ${file.name}...`);
    const response = await fetch(`${BACKEND_COMFYUI_URL}/upload/image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`Failed to upload ${file.name}`);
    return response.json();
  }, [updateStatus]);

  const getModifiedWorkflow = useCallback((workflow: any, destinationImageName: string, objectImageName: string, maskImageName: string | null) => {
    const updatedWorkflow = JSON.parse(JSON.stringify(workflow));

    // Different node titles based on selected accessory
    const destinationNodeTitle = selectedAccessory === 'watch' ? "Insert hand" : "Other insert";
    const objectNodeTitle = selectedAccessory === 'watch' ? "Insert object" : "Other insert object";
    const maskNodeTitle = selectedAccessory === 'watch' ? "Insert Mask" : "Load Other Mask";
    let destinationNodeId, objectNodeId, maskNodeId;
    for (const id in updatedWorkflow) {
      if (updatedWorkflow[id]._meta?.title === destinationNodeTitle) destinationNodeId = id;
      if (updatedWorkflow[id]._meta?.title === objectNodeTitle) objectNodeId = id;
      if (updatedWorkflow[id]._meta?.title == maskNodeTitle) maskNodeId = id;
    }

    if (destinationNodeId) updatedWorkflow[destinationNodeId].inputs.image = destinationImageName;
    if (objectNodeId) updatedWorkflow[objectNodeId].inputs.image = objectImageName;
    if (maskNodeId && maskImageName) {
      updatedWorkflow[maskNodeId].inputs.image = maskImageName;
    }
    return updatedWorkflow;
  }, [selectedAccessory]);

  const queuePrompt = useCallback((clientId: string, workflowObject: any) => {
    const payload = {
      client_id: clientId,
      prompt: workflowObject,
    };

    updateStatus('Queueing prompt...');
    return fetch(`${BACKEND_COMFYUI_URL}/prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }, [updateStatus]);

  const startGeneration = useCallback(async () => {
    if (!destinationImage || !objectImage) {
      updateStatus('Error: Please select both a destination and object image.', true);
      return;
    }
    const logImageDimensions = async () => {
      console.log("🔍 Checking dimensions of images to be uploaded...");

      const getImageDimensions = (file: File) => {
        return new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => {
            resolve({
              "Image Type": file.name.startsWith('mask_') ? 'Mask' : (file === destinationImage ? 'Destination' : 'Object'),
              "File Name": file.name,
              "Width": img.width,
              "Height": img.height,
              "Dimensions": `${img.width} x ${img.height}`,
            });
            // Clean up the object URL to prevent memory leaks
            URL.revokeObjectURL(img.src);
          };
          img.onerror = (err) => reject(err);
          img.src = URL.createObjectURL(file);
        });
      };

      try {
        const promises = [];
        if (destinationImage) promises.push(getImageDimensions(destinationImage));
        if (objectImage) promises.push(getImageDimensions(objectImage));
        if (maskImage) promises.push(getImageDimensions(maskImage));

        const dimensions = await Promise.all(promises);
        console.table(dimensions);
      } catch (e) {
        console.error("Could not read image dimensions for debugging.", e);
      }
    };
    await logImageDimensions();
    setIsGenerating(true);
    setResultImage(null); // Reset previous result
    setProgress(0);

    try {
      // *** NEW: Upload the mask image first if it exists ***
      let maskFileInfo = null;
      if (maskImage) {
        updateStatus(`Uploading mask...`);
        maskFileInfo = await uploadImage(maskImage);
      }

      const destFileInfo = await uploadImage(destinationImage);
      const objFileInfo = await uploadImage(objectImage);
      updateStatus('Uploads complete. Preparing workflow...');

      const selectedWorkflow = selectedAccessory === 'watch' ? WORKFLOW_JSON_1 : WORKFLOW_JSON_2;

      // *** MODIFIED: Pass the mask filename to the workflow function ***
      const modifiedWorkflow = getModifiedWorkflow(
        selectedWorkflow,
        destFileInfo.name,
        objFileInfo.name,
        maskFileInfo ? maskFileInfo.name : null
      );

      const clientId = Math.random().toString(36).substring(7);
      const wsUrl = BACKEND_COMFYUI_URL.replace(/^http/, 'ws');
      // ... rest of the try block
      const ws = new WebSocket(`${wsUrl}/ws?clientId=${clientId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        queuePrompt(clientId, modifiedWorkflow);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'executed') {
            const executedNodeId = data.data.node;

            // Dynamically find the ID of the node titled "Final Image"
            const currentWorkflow = selectedAccessory === 'watch' ? WORKFLOW_JSON_1 : WORKFLOW_JSON_2;
            const finalNodeId = Object.keys(currentWorkflow).find(
              id => (currentWorkflow as any)[id]._meta?.title === 'Final Output'
            );

            // Check if the executed node is our final target node
            if (executedNodeId === finalNodeId) {
              const output = data.data?.output;

              if (output && output.images && output.images.length > 0) {
                // SUCCESS!
                updateStatus('Execution complete! Fetching image...');
                const finalImage = output.images[0];
                const imageUrl = `${BACKEND_COMFYUI_URL}/view?filename=${encodeURIComponent(finalImage.filename)}&subfolder=${encodeURIComponent(finalImage.subfolder)}&type=${finalImage.type}`;

                setResultImage(imageUrl);
                updateStatus('Done!');
                setProgress(100);
                toast.success("Image generated successfully!");
              } else {
                // This case should ideally not be hit with the new logic
                updateStatus(`Error: Final node ${finalNodeId} executed, but no image was found in the output.`, true);
                console.error("The final data packet was received but could not be parsed correctly:", data);
              }

              ws.close(); // Close the connection after getting the result
            }
          } else if (data.type === 'progress') {
            const { value, max } = data.data;
            const progressValue = Math.round((value / max) * 100);
            setProgress(progressValue);
            updateStatus(`Generating... ${progressValue}%`);
          } else if (data.type === 'execution_error') {
            updateStatus(`Server Error: ${JSON.stringify(data.data)}`, true);
          }
        } catch (e) {
          console.error("Error processing WebSocket message:", e);
          console.error("The problematic raw message was:", event.data);
        }
      };

      ws.onclose = () => {
        setIsGenerating(false);
      };

      ws.onerror = (err) => {
        updateStatus('WebSocket error. Check console for details.', true);
        console.error('WebSocket Error:', err);
        setIsGenerating(false);
      };

    } catch (error: any) {
      updateStatus(`An error occurred: ${error.message}`, true);
      console.error(error);
      setIsGenerating(false);
      setProgress(0);
    }
  }, [destinationImage, objectImage, maskImage, selectedAccessory, uploadImage, getModifiedWorkflow, queuePrompt, updateStatus]);

  // Canvas setup and drawing functions
  const setupCanvas = useCallback((image: HTMLImageElement) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;

    const MAX_CANVAS_DIMENSION = 4096;
    let drawWidth = image.width;
    let drawHeight = image.height;

    if (drawWidth > MAX_CANVAS_DIMENSION || drawHeight > MAX_CANVAS_DIMENSION) {
      toast.info(`Image is very large, downscaling for editor performance...`);
      const ratio = Math.min(MAX_CANVAS_DIMENSION / drawWidth, MAX_CANVAS_DIMENSION / drawHeight);
      drawWidth = Math.floor(drawWidth * ratio);
      drawHeight = Math.floor(drawHeight * ratio);
    }
    
    // --- NEW: Store the calculated canvas dimensions in state ---
    setCanvasSize({ width: drawWidth, height: drawHeight });

    canvas.width = drawWidth;
    canvas.height = drawHeight;

    context.drawImage(image, 0, 0, drawWidth, drawHeight);

    contextRef.current = context;
    originalImageRef.current = image;
  }, []);

  const handleDestinationImageChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        // Create a canvas to draw the image, which strips EXIF data
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          toast.error("Could not process image.");
          return;
        }

        // The browser reads EXIF for .width and .height, so the orientation is correct here
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Create a new file from the canvas blob
        canvas.toBlob((blob) => {
          if (blob) {
            const sanitizedFile = new File([blob], file.name.replace(/(\.[\w\d_-]+)$/i, '.png'), {
              type: 'image/png',
            });

            // Set the new, sanitized file as the destination image
            setDestinationImage(sanitizedFile);
            setMaskImage(null);
            setEditedImageData(null);

            toast.success("Image processed and ready.");
          }
        }, 'image/png');
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  }, []);

  const openEditor = useCallback(() => {
    if (!destinationImage) {
      toast.error('Please select a destination image first');
      return;
    }

    setIsEditorOpen(true);

    // Small delay to ensure dialog is open and canvas is rendered
    setTimeout(() => {
      const img = new Image();
      img.onload = () => {
        setupCanvas(img);
      };
      img.onerror = () => {
        toast.error('Failed to load image for editing');
      };
      img.src = URL.createObjectURL(destinationImage);
    }, 100);
  }, [destinationImage, setupCanvas]);

  const getBrushPosition = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    // This calculates the correct mouse position relative to the canvas,
    // accounting for any scaling or resizing of the canvas element in the CSS.
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }, []);

  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !contextRef.current || !canvasRef.current || !originalImageRef.current) return;

    const context = contextRef.current;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    // --- FIX: Add a guard to prevent division by zero ---
    // This stops the function if the canvas has no rendered size yet.
    if (rect.width === 0 || rect.height === 0) {
      return;
    }
    // --- End of Fix ---

    const scale = canvas.width / rect.width;
    const scaledBrushSize = brushSize * scale;

    const { x, y } = getBrushPosition(e);

    if (brushColor === 'white') { // Erase logic
      context.globalCompositeOperation = 'destination-out';
      context.fillStyle = '#000';

      context.beginPath();
      if (brushType === 'round') {
        context.arc(x, y, scaledBrushSize / 2, 0, 2 * Math.PI);
      } else {
        context.rect(x - scaledBrushSize / 2, y - scaledBrushSize / 2, scaledBrushSize, scaledBrushSize);
      }
      context.fill();

    } else { // Restore logic
      context.globalCompositeOperation = 'source-over';

      context.save();
      context.beginPath();
      if (brushType === 'round') {
        context.arc(x, y, scaledBrushSize / 2, 0, 2 * Math.PI);
      } else {
        context.rect(x - scaledBrushSize / 2, y - scaledBrushSize / 2, scaledBrushSize, scaledBrushSize);
      }
      context.clip();
      context.drawImage(originalImageRef.current, 0, 0);
      context.restore();
    }
  }, [isDrawing, brushColor, brushSize, brushType, getBrushPosition]);

  const startDrawing = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!contextRef.current) return;
    setIsDrawing(true);
    draw(e);
  }, [draw]);


  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
    // IMPORTANT: Reset composite operation to default after drawing is finished.
    if (contextRef.current) {
      contextRef.current.globalCompositeOperation = 'source-over';
    }
    contextRef.current?.beginPath();
  }, []);

  const resetCanvas = useCallback(() => {
    if (!originalImageRef.current) return;
    setupCanvas(originalImageRef.current);
  }, [setupCanvas]);

  const saveEditedImage = useCallback(() => {
    if (!canvasRef.current || !originalImageRef.current || !destinationImage) {
      toast.error("Cannot save, required image data is missing.");
      return;
    }

    // For the UI preview, we still want to show the version with transparency
    const previewDataUrl = canvasRef.current.toDataURL('image/png');
    // Make sure you have a state setter like setEditedImageData or setEditedImageDataUrl
    // For this example, I'll assume it's called setEditedImageData
    if (typeof setEditedImageData !== 'undefined') {
      setEditedImageData(previewDataUrl);
    }


    // --- Start of Final, Pixel-by-Pixel Saving Logic ---

    const maskCanvas = document.createElement('canvas');
    const maskCtx = maskCanvas.getContext('2d');
    if (!maskCtx) {
      toast.error("Could not create mask canvas.");
      return;
    }

    const width = canvasRef.current.width;
    const height = canvasRef.current.height;
    maskCanvas.width = width;
    maskCanvas.height = height;

    const editedCtx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!editedCtx) return;
    const editedImageData = editedCtx.getImageData(0, 0, width, height);
    const editedData = editedImageData.data;

    const maskOutputImageData = maskCtx.createImageData(width, height);
    const maskData = maskOutputImageData.data;

    for (let i = 0; i < editedData.length; i += 4) {
      // If the pixel in the edited canvas is transparent, it's the masked area.
      // In our mask file, the masked area should be WHITE.
      if (editedData[i + 3] < 128) {
        maskData[i] = 255; maskData[i + 1] = 255; maskData[i + 2] = 255; maskData[i + 3] = 255;
      } else {
        maskData[i] = 0; maskData[i + 1] = 0; maskData[i + 2] = 0; maskData[i + 3] = 255;
      }
    }

    maskCtx.putImageData(maskOutputImageData, 0, 0);

    maskCanvas.toBlob((blob) => {
      if (blob) {
        const maskFile = new File([blob], `mask_${destinationImage.name.replace(/(\.[\w\d_-]+)$/i, '.png')}`, { type: 'image/png' });
        setMaskImage(maskFile);
        toast.success('Mask created successfully!');
        setIsEditorOpen(false);
      } else {
        toast.error('Failed to create mask file.');
      }
    }, 'image/png');
  }, [destinationImage]);

  const getCursorStyle = useCallback(() => {
    const size = Math.max(brushSize, 8);
    let cursorSvg;

    if (brushType === 'square') {
      // --- NEW: Generate an SVG for a square cursor ---
      cursorSvg = `url("data:image/svg+xml,%3csvg width='${size}' height='${size}' xmlns='http://www.w3.org/2000/svg'%3e%3crect x='1' y='1' width='${size - 2}' height='${size - 2}' fill='none' stroke='%23000' stroke-width='1'/%3e%3c/svg%3e") ${size / 2} ${size / 2}, crosshair`;
    } else {
      // Default to the round cursor
      cursorSvg = `url("data:image/svg+xml,%3csvg width='${size}' height='${size}' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='${size / 2}' cy='${size / 2}' r='${size / 2 - 1}' fill='none' stroke='%23000' stroke-width='1'/%3e%3c/svg%3e") ${size / 2} ${size / 2}, crosshair`;
    }

    return {
      cursor: cursorSvg
    };
    // --- FIX: Add brushType to the dependency array ---
  }, [brushSize, brushType]);
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();

    const scrollDelta = e.deltaY > 0 ? -0.15 : 0.15; // Zoom out on scroll down, in on up

    // --- FIX #2: Dynamically calculate max zoom to prevent crashes ---
    const maxDim = Math.max(canvasSize.width, canvasSize.height);
    // Keep the effective rendered size below a safe browser limit (e.g., 8192px)
    const maxZoom = maxDim > 0 ? Math.max(1.0, 8192 / maxDim) : 3; // Default max 3x zoom

    const newZoom = Math.max(0.2, Math.min(maxZoom, zoom + scrollDelta));

    const canvasContainer = e.currentTarget;
    const rect = canvasContainer.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // --- FIX #1: Corrected formula for centered zoom ---
    const newOffsetX = mouseX - (mouseX - offset.x) * (newZoom / zoom);
    const newOffsetY = mouseY - (mouseY - offset.y) * (newZoom / zoom);

    setZoom(newZoom);
    setOffset({ x: newOffsetX, y: newOffsetY });

  }, [zoom, offset, canvasSize]); // Update dependencies

  const panStart = useCallback((e: React.MouseEvent) => {
    // Pan starts on middle mouse button click
    if (e.button !== 1) return;
    e.preventDefault();
    isPanning.current = true;
    lastPanPoint.current = { x: e.clientX, y: e.clientY };
    (e.currentTarget as HTMLElement).style.cursor = 'grabbing';
  }, []);

  const panMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning.current) return;
    const dx = e.clientX - lastPanPoint.current.x;
    const dy = e.clientY - lastPanPoint.current.y;
    setOffset(prev => ({ x: prev.x + dx, y: prev.y + dy }));
    lastPanPoint.current = { x: e.clientX, y: e.clientY };
  }, []);

  const panEnd = useCallback((e: React.MouseEvent) => {
    isPanning.current = false;
    (e.currentTarget as HTMLElement).style.cursor = 'default';
  }, []);

  const resetView = useCallback(() => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  return (
    <div className="space-y-6">
      {/* Accessory Selection at the top */}
      <Card className="bg-slate-00/50 border-slate-700">
        <CardContent className="p-6">
          <div>
            <Label htmlFor="accessory-select" className="text-white font-medium">
              Select Accessory Type
            </Label>
            <select
              id="accessory-select"
              value={selectedAccessory}
              onChange={(e) => setSelectedAccessory(e.target.value)}
              className="mt-2 w-full px-3 py-2 bg-[#A3E635]/80 hover:bg-green-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            >
              <option value="">--Select an accessory--</option>
              <option value="watch">Watch</option>
              <option value="cap">Cap</option>
              <option value="bracelet">Bracelet</option>
            </select>
            <p className="text-sm text-white mt-1">
              {selectedAccessory === 'watch'
                ? 'Using specialized watch insertion workflow'
                : 'Using general accessory insertion workflow'}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* --- RESPONSIVE Three column layout: Stacks on mobile, horizontal on desktop --- */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Destination Image */}
        <Card className="bg-slate-00/50 border-slate-700 flex flex-col">
          <CardContent className="p-6 space-y-4 flex-grow flex flex-col">
            <div>
              <Label htmlFor="destination-image" className="text-foreground font-medium">
                {selectedAccessory === 'watch' ? 'Hand Image (with mask)' : 'Destination Image (with mask)'}
              </Label>
              <Input
                id="destination-image"
                type="file"
                accept="image/*"
                onChange={handleDestinationImageChange}
                className="mt-2 bg-[#A3E635]/80 hover:bg-green-700 border-border/20"
              />
              <p className="text-sm text-foreground/60 mt-1">
                {selectedAccessory === 'watch'
                  ? 'Upload an image of a hand where you want to place the watch'
                  : `Upload an image where you want to place the ${selectedAccessory}`}
              </p>
            </div>

            {destinationImage && (
              <div className="space-y-4 mt-auto">
                <div className="flex items-center justify-between">
                  <div className="flex gap-2">
                    <Button
                      onClick={openEditor}
                      variant="default"
                      size="sm"
                    >
                      Edit Mask
                    </Button>
                  </div>
                </div>

                {/* --- RESPONSIVE Image Preview Container --- */}
                <div className="relative w-full h-72 lg:h-96 bg-black/20 rounded-lg flex items-center justify-center">
                  <img
                    src={editedImageData || URL.createObjectURL(destinationImage)}
                    alt="Destination"
                    className="max-w-full max-h-full object-contain"
                  />
                  {editedImageData && (
                    <div className="absolute top-2 left-2 bg-green-500/80 text-white px-2 py-1 rounded text-xs">
                      Edited
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Middle column - Object Image */}
        <Card className="bg-slate-00/50 border-slate-700 flex flex-col">
          <CardContent className="p-6 space-y-4 flex-grow flex flex-col">
            <div>
              <Label htmlFor="object-image" className="text-foreground font-medium">
                {selectedAccessory.charAt(0).toUpperCase() + selectedAccessory.slice(1)} to Insert
              </Label>
              <Input
                id="object-image"
                type="file"
                accept="image/*"
                onChange={(e) => setObjectImage(e.target.files?.[0] || null)}
                className="mt-2 bg-[#A3E635]/80 hover:bg-green-700 border-border/20"
              />
              <p className="text-sm text-foreground/60 mt-1">
                Upload an image of the {selectedAccessory} you want to insert
              </p>
            </div>
            
            {objectImage && (
              <div className="relative mt-auto w-full h-72 lg:h-96 bg-black/20 rounded-lg flex items-center justify-center">
                <img
                  src={URL.createObjectURL(objectImage)}
                  alt="Object Preview"
                  className="max-w-full max-h-full object-contain"
                />
                <div className="absolute top-2 left-2 bg-blue-500/80 text-white px-2 py-1 rounded text-xs font-medium">
                  Object Preview
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right column - Result */}
        <Card className="bg-slate-00/50 border-slate-700 flex flex-col">
          <CardContent className="p-6 flex flex-col h-full">
            <div className="flex-1 flex flex-col">
              <Label className="text-foreground font-medium mb-4">Result</Label>
              <div className="relative w-full h-72 lg:h-96 bg-black/20 rounded-lg flex items-center justify-center flex-grow">
                {resultImage ? (
                  <img
                    src={resultImage}
                    alt="Generated"
                    className="max-w-full max-h-full object-contain"
                  />
                ) : (
                  <div className="text-center text-foreground/60">
                    <p>Generated image</p>
                    <p>will appear here</p>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-4 mt-4">
              <div className="text-sm text-foreground/80 font-mono whitespace-pre-wrap bg-muted/20 px-3 py-1 rounded-md border border-border/20 h-16 overflow-y-auto">
                {status}
              </div>
              {progress > 0 && (
                <div className="w-full bg-muted/20 rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
              <Button
    onClick={startGeneration}
    disabled={isGenerating || !destinationImage || !objectImage}
    className={`w-full py-3 text-white font-medium
      transition-all duration-300 hover:scale-105 active:scale-95
      relative flex items-center justify-center
      ${isGenerating
        ? 'bg-[#A3E635]/90'
        : 'bg-[#A3E635]/80 hover:bg-green-700'
      }
    `}
>
    {isGenerating ? (
        <>
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
            <span className="animate-pulse">Processing...</span>
        </>
    ) : (
        'Generate Image'
    )}
</Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Image Editor Modal (No changes needed here, it's already responsive) */}
      <Dialog open={isEditorOpen} onOpenChange={setIsEditorOpen}>
        <DialogContent className="w-[95vw] max-w-4xl h-[90vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>Edit Image Mask</DialogTitle>
          </DialogHeader>
          <div className="flex-grow flex flex-col space-y-4 min-h-0">
            {/* Editor Controls */}
            <div className="flex flex-wrap items-center gap-4 p-2 bg-muted/20 rounded-lg border border-border/20">
              <div className="flex gap-2">
                <Button variant={brushType === 'round' ? 'default' : 'outline'} size="sm" onClick={() => setBrushType('round')}>Round</Button>
                <Button variant={brushType === 'square' ? 'default' : 'outline'} size="sm" onClick={() => setBrushType('square')}>Square</Button>
              </div>
              <div className="flex gap-2">
                <Button variant={brushColor === 'black' ? 'default' : 'outline'} size="sm" onClick={() => setBrushColor('black')}>Erase</Button>
                <Button variant={brushColor === 'white' ? 'default' : 'outline'} size="sm" onClick={() => setBrushColor('white')}>Mask</Button>
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-sm">Brush Size: {brushSize}px</Label>
                <div className="w-32"><Slider value={[brushSize]} onValueChange={(value) => setBrushSize(value[0])} max={50} min={1} step={1} /></div>
                <Input type="number" value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} min={1} max={50} className="w-20 h-8" />
              </div>
              <Button onClick={resetCanvas} variant="outline" size="sm">Reset Mask</Button>
              <Button onClick={resetView} variant="outline" size="sm">Reset View</Button>
            </div>
            
            {/* Canvas Container */}
            <div className="w-full flex-grow relative bg-muted/10 rounded-lg" style={{ overflow: 'hidden', cursor: 'default' }} onWheel={handleWheel} onMouseDown={panStart} onMouseMove={panMove} onMouseUp={panEnd} onMouseLeave={panEnd}>
              <div className="absolute inset-0 flex items-center justify-center p-2">
                <canvas
                  ref={canvasRef}
                  onMouseDown={startDrawing}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                  className="border border-border/20"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
                    transformOrigin: '0 0',
                    ...getCursorStyle()
                  }}
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" onClick={() => setIsEditorOpen(false)}>Cancel</Button>
              <Button onClick={saveEditedImage} className="bg-green-600 hover:bg-green-700">Save Changes</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};
