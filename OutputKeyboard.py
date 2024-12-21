import cv2
import numpy as np

KEY_COORDINATES = {
    "C": 10,
    "C#": 85,
    "D": 50,
    "D#": 170,
    "E": 90,
    "F": 130,
    "F#": 340,
    "G": 170,
    "G#": 430,
    "A": 210,
    "A#": 510,
    "B": 250,
}

class virtualKeyboard:
    def __init__(self, base_path='./KeyImages/smallKey.png', overlay_path='./KeyImages/key_green_top.png'):
        # Load the base keyboard image in BGR
        self.base_img = cv2.imread(base_path, cv2.IMREAD_COLOR)
        if self.base_img is None:
            raise FileNotFoundError(f"Base image not found: {base_path}")
        self.base_img = cv2.resize(self.base_img, (600,400))

        # Load the overlay image with alpha channel (BGRA)
        overlay_img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            raise FileNotFoundError(f"Overlay image not found: {overlay_path}")

        # Resize overlay if needed (50x400 as previously mentioned)
        self.overlay_img = cv2.resize(overlay_img, (50, 400))

    def draw_highlighted_keys(self, notes):
        # Make a copy of the base image so we don't modify the original
        result_img = self.base_img.copy()

        # For each note, paste overlay onto result_img at given coordinates
        for note in notes:
            if note not in KEY_COORDINATES:
                print(f"Warning: No coordinate defined for note '{note}'")
                continue

            x = KEY_COORDINATES[note]
            y = 0  # top-left corner

            h, w = self.overlay_img.shape[:2]
            if x + w > result_img.shape[1] or y + h > result_img.shape[0]:
                print("part1")
                print(x + w > result_img.shape[1])
                print( y + h > result_img.shape[0])
                print(f"Warning: Overlay for note '{note}' out of image bounds, skipping.")
                continue

            # Extract overlay regions
            overlay_region = self.overlay_img[..., :3]  # BGR channels
            alpha_channel = self.overlay_img[..., 3] / 255.0  # Normalize alpha to [0,1]

            # Get region of interest from the result image
            roi = result_img[y:y+h, x:x+w]

            # Perform alpha blending
            for c in range(3):
                roi[..., c] = (alpha_channel * overlay_region[..., c] +
                               (1 - alpha_channel) * roi[..., c])

            # Place blended region back
            result_img[y:y+h, x:x+w] = roi

        return result_img

