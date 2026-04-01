"""
morphology.py - THE CRISP CHARACTER VERSION
------------------------------------------
Optimized to preserve individual characters for OCR.
Removes aggressive smearing and destructive border clearing.
"""

import cv2
import numpy as np
from typing import Tuple

def process_plate_region(gray_image: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         ed_kernel: int = 3,
                         closing_kernel: int = 5
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Refined morphological pipeline for perfect OCR.
    Isolates characters without melting them together.
    """
    x, y, w, h = bbox
    # Ensure coordinates are within image bounds
    y = max(0, y); x = max(0, x)
    roi_gray = gray_image[y:y+h, x:x+w]

    if roi_gray.size == 0:
        return roi_gray, roi_gray, roi_gray, roi_gray, roi_gray

    # 1. PREP: Light Blur to remove salt/pepper noise before thresholding
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    # 2. OTSU'S BINARIZATION: Mathematically finds the perfect shadow threshold.
    # We use BINARY_INV so the text becomes White and the background becomes Black.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. SAFE BORDER CLEARING:
    # Instead of flood-fill that eats letters, we just draw a 2-pixel black
    # rectangle around the very edge to erase the physical plastic plate frame.
    border_cleared = binary.copy()
    cv2.rectangle(border_cleared, (0, 0), (w-1, h-1), 0, 2)

    # 4. LIGHT MORPHOLOGY:
    # Use small 2x2 kernels. We are treating this with a scalpel, not a hammer.

    # Opening: Removes tiny white specs (noise) in the black background
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(border_cleared, cv2.MORPH_OPEN, kernel_open)

    # Closing: Fills in tiny black holes inside the solid white letters
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    final_clean = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Return the 5 steps mapped to viewer.py's visual strip
    return roi_gray, binary, border_cleared, opened, final_clean