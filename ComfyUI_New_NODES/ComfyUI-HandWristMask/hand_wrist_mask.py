import torch
import numpy as np
import cv2

# Try to import mediapipe and notify the user if it's not installed.
try:
    import mediapipe as mp
except ImportError:
    print("--------------------------------------------------------------------")
    print("Warning: Mediapipe not installed for ComfyUI-HandWristMask node.")
    print("Please install it using: pip install mediapipe")
    print("--------------------------------------------------------------------")

class HandWristMask:
    """
    A ComfyUI node that detects a hand and creates an intelligently oriented mask
    over the wrist, suitable for inpainting a watch. This version uses a robust
    method to handle various hand orientations (front-facing, side-facing).
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node. The factors are now based on the
        more stable palm-length calculation.
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "width_proxy_factor": ("FLOAT", {"default": 1.1, "min": 0.1, "max": 3.0, "step": 0.05}),
                "height_factor": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 3.0, "step": 0.05}),
                "shift_factor": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE",)
    RETURN_NAMES = ("mask", "image",)
    FUNCTION = "create_wrist_mask"
    CATEGORY = "Image/Masking"

    def create_wrist_mask(self, image, width_proxy_factor, height_factor, shift_factor):
        """
        The core logic of the node, updated with the more robust orientation and
        sizing calculation provided by the user.
        """
        output_masks = []
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            for img_tensor in image:
                img_np = np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                h, w, _ = img_np.shape

                mask_np = np.zeros((h, w), dtype=np.uint8)
                results = hands.process(img_np) # Image is already RGB

                if results.multi_hand_landmarks:
                    print("HandWristMask Node: Hand detected! Creating mask with robust logic.")
                    landmarks = results.multi_hand_landmarks[0].landmark

                    def get_coords(landmark_index):
                        point = landmarks[landmark_index]
                        return np.array([point.x * w, point.y * h], dtype=np.float32)

                    # 1. Use stable landmarks for direction (wrist and middle finger base)
                    p_wrist = get_coords(mp_hands.HandLandmark.WRIST)
                    p_middle_mcp = get_coords(mp_hands.HandLandmark.MIDDLE_FINGER_MCP)

                    # 2. Calculate a stable direction vector and palm length
                    v_dir = p_middle_mcp - p_wrist
                    palm_length = np.linalg.norm(v_dir) + 1e-6 # Add epsilon for safety
                    v_dir = v_dir / palm_length # Normalize

                    # 3. Calculate perpendicular vector and infer dimensions from palm length
                    v_perp = np.array([-v_dir[1], v_dir[0]])
                    wrist_width = palm_length * width_proxy_factor
                    wrist_height = wrist_width * height_factor

                    # 4. Calculate the four corners of the polygon
                    p1_base = p_wrist - v_perp * wrist_width / 2
                    p2_base = p_wrist + v_perp * wrist_width / 2
                    p3_base = p2_base - v_dir * wrist_height
                    p4_base = p1_base - v_dir * wrist_height
                    
                    # 5. Shift the mask down the arm
                    shift_vector = -v_dir * wrist_height * shift_factor
                    
                    p1 = p1_base + shift_vector
                    p2 = p2_base + shift_vector
                    p3 = p3_base + shift_vector
                    p4 = p4_base + shift_vector

                    wrist_polygon = np.array([p1, p2, p3, p4], dtype=np.int32)

                    # 6. Draw the final polygon on the mask
                    cv2.fillPoly(mask_np, [wrist_polygon], 255)
                
                else:
                    print("HandWristMask Node: No hand detected. Returning an empty mask.")

                # Convert the final numpy mask back to a torch tensor
                mask = torch.from_numpy(mask_np).to(torch.float32) / 255.0
                output_masks.append(mask)

        masks_tensor = torch.stack(output_masks)
        return (masks_tensor, image,)
