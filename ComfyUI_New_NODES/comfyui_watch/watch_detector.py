import cv2
import numpy as np
import torch

class WatchDetector:
    """
    A ComfyUI custom node to detect a circular watch, generate a mask, 
    and return the watch isolated on a solid color background.
    This node uses OpenCV's Hough Circle Transform with resolution-independent parameters.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node. Radius and distance parameters
        are now factors of the image size for resolution independence.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "dp": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 5.0, "step": 0.1}),
                "param1": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 500.0, "step": 1.0}),
                "param2": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 500.0, "step": 1.0}),
                # These factors are relative to the smaller dimension of the image.
                "min_dist_factor": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_radius_factor": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_radius_factor": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 1.0, "step": 0.01}),
                "bg_red": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
                "bg_green": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
                "bg_blue": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("mask", "image_on_bg",)
    FUNCTION = "detect_watch"

    CATEGORY = "image"

    def detect_watch(self, image, dp, param1, param2, min_dist_factor, min_radius_factor, max_radius_factor, bg_red, bg_green, bg_blue):
        """
        This function isolates the detected watch on a custom solid color background.
        It calculates pixel-based parameters from relative factors to handle any image size.
        """
        # Get image dimensions from the tensor (batch, height, width, channels)
        height, width = image.shape[1], image.shape[2]
        smaller_dim = min(height, width)

        # --- Calculate absolute pixel values from relative factors ---
        min_dist = int(smaller_dim * min_dist_factor)
        min_radius = int(smaller_dim * min_radius_factor)
        max_radius = int(smaller_dim * max_radius_factor)

        # Convert the input tensor image to a NumPy array for OpenCV processing.
        img_np = np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8)
        
        # Convert to grayscale and blur to prepare for circle detection.
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Detect circles using the dynamically calculated pixel values.
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
        )
        
        # Create a black mask to draw the detected circle on.
        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            largest_circle = max(circles[0, :], key=lambda x: x[2])
            center = (largest_circle[0], largest_circle[1])
            radius = largest_circle[2]
            cv2.circle(mask, center, radius, (255, 255, 255), -1)

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
