import cv2
import numpy as np

TARGET_WIDTH = 620


def preprocess(image: np.ndarray, target_width: int = TARGET_WIDTH, **kwargs):
    # 1. Resize
    h, w = image.shape[:2]
    scale = target_width / w
    resized = cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Bilateral Filter (DO THIS FIRST)
    # We smooth the noise while the image is still "calm"
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. SMART NORMALIZATION (Replaces Sharpening)
    # This stretches the contrast so the darkest pixel is 0 and brightest is 255
    # without creating the "halos" that broke your white plates.
    norm_img = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 5. SUBTLE CLAHE
    # ClipLimit 2.0 is the "sweet spot" for white plates.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    final_gray = clahe.apply(norm_img)

    return resized, gray, final_gray, scale