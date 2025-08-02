import kornia
import torch
from paddleocr import PaddleOCR
import comfy.model_management


class OcrBoxMask:
    def __init__(self):
        print("OcrFunction init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {"required":
            {
                "lang": (lang_list, {"default": "ch"}),
                "images": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("mask", "masked_image",)
    FUNCTION = 'orc_box_mask'

    def orc_box_mask(self, images, text, lang):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        
        masks = []
        masked_images = []

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            shape = i.shape
            mask = torch.zeros((shape[0], shape[1]), dtype=torch.uint8)
            
            processed_text = text.replace("\n", ";")
            words = [w.strip() for w in processed_text.split(";")]
            
            result = self.ocr.ocr(i, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        for word in words:
                            if word == "":
                                continue
                            if text == "" or line[1][0].find(word) >= 0:
                                text_line = line[1][0]
                                points = line[0]
                                total_length = len(text_line)
                                start = 0
                                while text_line.find(word, start) >= 0:
                                    start = text_line.find(word, start)
                                    end = start + len(word)
                                    x_min = points[0][0] + start * (points[1][0] - points[0][0]) / total_length
                                    x_max = points[0][0] + end * (points[1][0] - points[0][0]) / total_length
                                    y_top = min(points[0][1], points[1][1])
                                    y_bottom = max(points[2][1], points[3][1])
                                    
                                    # MODIFIED: Padding is now 5 pixels
                                    padding = 5
                                    y_top_padded = max(0, int(y_top) - padding)
                                    y_bottom_padded = min(shape[0], int(y_bottom) + padding)
                                    x_min_padded = max(0, int(x_min) - padding)
                                    x_max_padded = min(shape[1], int(x_max) + padding)

                                    mask[y_top_padded:y_bottom_padded, x_min_padded:x_max_padded] = 1
                                    start = end
            
            masked_image = image * mask.unsqueeze(-1)
            masked_images.append(masked_image.unsqueeze(0))
            masks.append(mask.unsqueeze(0))

        return (torch.cat(masks, dim=0), torch.cat(masked_images, dim=0),)


class OcrImageText:
    def __init__(self):
        print("OcrImageText init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {
            "required": {
                "images": ("IMAGE",),
                "lang": (lang_list, {"default": "ch"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text", "prompt",)
    FUNCTION = 'orc_image_text'

    def orc_image_text(self, images, lang):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

        final_text = ""
        prompt = ""
        last_text = ""
        is_first_image = True 

        for image in images:
            i = 255. * image.cpu().numpy()
            now_text = ""
            
            upper_texts = []
            lower_texts = []
            image_center_y = i.shape[0] / 2
            
            orc_ret = self.ocr.ocr(i, cls=False)
            for idx in range(len(orc_ret)):
                res = orc_ret[idx]
                if res is not None:
                    for line in res:
                        text_line = line[1][0]
                        if text_line != "":
                            now_text += text_line + "\n"
                            if is_first_image:
                                box_coords = line[0]
                                box_center_y = (box_coords[0][1] + box_coords[2][1]) / 2
                                if box_center_y < image_center_y:
                                    upper_texts.append(text_line)
                                else:
                                    lower_texts.append(text_line)
            
            if now_text != "" and now_text != last_text:
                final_text += now_text
                last_text = now_text
            
            if is_first_image and (upper_texts or lower_texts):
                brand_text = " ".join(upper_texts)
                extra_text = " ".join(lower_texts)
                prompt = f'"{brand_text}" text is on the above part of watch and "{extra_text}" is on the below part of the watch, in high quality in the mask'
                is_first_image = False 

        return (final_text, prompt,)


class OcrBlur:
    def __init__(self):
        print("OcrBlur init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {"required":
            {
                "lang": (lang_list, {"default": "ch"}),
                "images": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
                "blur": ("INT", {"default": 255, "min": 3, "max": 8191, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'orc_blur'

    def orc_blur(self, images, text, lang, blur):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        new_images = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            shape = i.shape
            mask = torch.zeros((shape[0], shape[1]), dtype=torch.uint8)
            
            processed_text = text.replace("\n", ";")
            words = [w.strip() for w in processed_text.split(";")]
            
            result = self.ocr.ocr(i, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        for word in words:
                            if word == "":
                                continue
                            if text == "" or line[1][0].find(word) >= 0:
                                text_line = line[1][0]
                                points = line[0]
                                total_length = len(text_line)
                                start = 0
                                while text_line.find(word, start) >= 0:
                                    start = text_line.find(word, start)
                                    end = start + len(word)
                                    x_min = points[0][0] + start * (points[1][0] - points[0][0]) / total_length
                                    x_max = points[0][0] + end * (points[1][0] - points[0][0]) / total_length
                                    y_top = min(points[0][1], points[1][1])
                                    y_bottom = max(points[2][1], points[3][1])
                                    
                                    # MODIFIED: Padding is now 5 pixels
                                    padding = 5
                                    y_top_padded = max(0, int(y_top) - padding)
                                    y_bottom_padded = min(shape[0], int(y_bottom) + padding)
                                    x_min_padded = max(0, int(x_min) - padding)
                                    x_max_padded = min(shape[1], int(x_max) + padding)

                                    mask[y_top_padded:y_bottom_padded, x_min_padded:x_max_padded] = 1
                                    start = end

            # blur the image by mask
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.permute(0, 3, 1, 2)
            blurred = image.clone()
            alpha = mask_floor(mask_unsqueeze(mask))
            alpha = alpha.expand(-1, 3, -1, -1)
            blurred = gaussian_blur(blurred, blur, 0)
            blurred = image + (blurred - image) * alpha
            new_images.append(blurred.permute(0, 2, 3, 1))
        return (torch.cat(new_images, dim=0),)


def gaussian_blur(image, radius: int, sigma: float = 0):
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    image = image.to(comfy.model_management.get_torch_device())
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma)).cpu()


def mask_floor(mask, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)


def mask_unsqueeze(mask):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask