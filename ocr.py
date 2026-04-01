"""
ocr.py
------
Handles Optical Character Recognition for the cleaned plate region.
V2 - Enhanced for better Tesseract accuracy.
"""

import cv2
import pytesseract
import numpy as np

# UPDATE THIS PATH to where you installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recognize_plate(binary_roi: np.ndarray) -> str:
    """
    Converts a binary plate image into a string of text.
    Includes upscaling, padding, and smart color inversion.
    """
    if binary_roi is None or binary_roi.size == 0:
        return ""

    # --- 1. SMART COLOR FLIP ---
    # Instead of a 5x5 corner, we check the entire outer border of the image.
    # If the border is mostly dark, the background is black, so we invert it.
    top, bottom = binary_roi[0, :], binary_roi[-1, :]
    left, right = binary_roi[:, 0], binary_roi[:, -1]
    border_pixels = np.concatenate([top, bottom, left, right])

    if np.mean(border_pixels) < 128:
        # Background is dark, invert to make it black text on white background
        processed_roi = cv2.bitwise_not(binary_roi)
    else:
        # Already black text on white background
        processed_roi = binary_roi.copy()

    # --- 2. RESOLUTION SWEET SPOT ---
    # Tesseract likes letters to be tall. We scale the height up to 70 pixels
    # and adjust the width to keep the proportions exactly the same.
    h, w = processed_roi.shape[:2]
    target_h = 70
    scale = target_h / float(h)
    target_w = int(w * scale)
    processed_roi = cv2.resize(processed_roi, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # --- 3. THE CLAUSTROPHOBIA FIX (PADDING) ---
    # Add a 20-pixel solid white border around the whole image
    # This acts as the "page margins" Tesseract needs to read properly.
    pad = 20
    processed_roi = cv2.copyMakeBorder(processed_roi, pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # --- 4. TESSERACT CONFIG ---
    # --psm 7: Treat the image as a single text line
    # --oem 3: Default OCR engine
    # whitelist: Only allow capital letters and numbers
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    try:
        text = pytesseract.image_to_string(processed_roi, config=custom_config)
        # Strip out any accidental spaces or newlines Tesseract hallucinated
        clean_text = "".join(text.split())
        return clean_text
    except Exception as e:
        return f"Error: {str(e)}"