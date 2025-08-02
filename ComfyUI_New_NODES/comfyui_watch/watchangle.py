import cv2
import numpy as np
import torch
import math

class WatchAngle:
    """
    A ComfyUI custom node to detect a circular or elliptical watch, generate a mask, 
    and return the watch isolated on a solid color background.
    This node uses contour detection and ellipse fitting for robust, angled detection.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node. Parameters are now for Canny edge
        detection and contour filtering to allow for ellipse detection.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "canny_threshold1": ("FLOAT", {"default": 50, "min": 0, "max": 255, "step": 1}),
                "canny_threshold2": ("FLOAT", {"default": 150, "min": 0, "max": 255, "step": 1}),
                "min_contour_points": ("INT", {"default": 15, "min": 5, "max": 200, "step": 1}),
                "bg_red": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
                "bg_green": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
                "bg_blue": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("mask", "image_on_bg",)
    FUNCTION = "detect_watch"

    CATEGORY = "image"

    def detect_watch(self, image, canny_threshold1, canny_threshold2, min_contour_points, bg_red, bg_green, bg_blue):
        """
        This function isolates the detected watch on a custom solid color background
        using an ellipse-fitting method.
        """
        # Convert the input tensor image to a NumPy array for OpenCV processing.
        img_np = np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
        
        # --- Ellipse Detection Pipeline ---
        # 1. Convert to grayscale and apply a Gaussian blur to reduce noise.
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Use Canny edge detector to find edges in the image.
        edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
        
        # 3. Find contours from the edge-detected image.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Filter contours and fit ellipses.
        found_ellipses = []
        for contour in contours:
            # A contour needs at least 5 points to fit an ellipse.
            if len(contour) >= min_contour_points:
                # Fit an ellipse to the contour.
                ellipse = cv2.fitEllipse(contour)
                found_ellipses.append(ellipse)

        # Create a black mask to draw the detected ellipse on.
        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

        # 5. Find the largest ellipse and draw it on the mask.
        if found_ellipses:
            # Find the ellipse with the largest area (width * height).
            largest_ellipse = max(found_ellipses, key=lambda e: e[1][0] * e[1][1])
            cv2.ellipse(mask, largest_ellipse, (255, 255, 255), -1)

        # --- Compositing ---
        # Convert the NumPy mask to a PyTorch tensor for output.
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_for_output = mask_tensor.unsqueeze(0)
        
        # Move the mask to the same device as the input image tensor.
        mask_on_device = mask_for_output.to(image.device)
        
        # Prepare mask for compositing by adding a channel dimension.
        mask_for_image_op = mask_on_device.unsqueeze(-1).expand_as(image)

        # Create the solid color background.
        norm_r, norm_g, norm_b = bg_red / 255.0, bg_green / 255.0, bg_blue / 255.0
        bg_color_tensor = torch.tensor([norm_r, norm_g, norm_b]).to(image.device)
        background = torch.full_like(image, 0)
        background[:, :, :, :] = bg_color_tensor

        # Composite the final image.
        final_image = image * mask_for_image_op + background * (1 - mask_for_image_op)

        return (mask_for_output, final_image,)
