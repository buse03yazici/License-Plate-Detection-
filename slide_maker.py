"""
slide_maker.py - The Presentation Generator
-------------------------------------------
Automatically generates a high-resolution, 16:9 PowerPoint-ready slide
for EVERY image in your dataset. No typing paths required.
"""

import cv2
import os
import glob
import numpy as np
import pytesseract
import matplotlib
matplotlib.use("Agg")  # Prevents plotting window from crashing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from preprocessing import preprocess
from segmentation  import detect_plate
from morphology    import process_plate_region

# Point this to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def run_ocr(binary_roi):
    if binary_roi is None or binary_roi.size == 0: return "N/A"
    inverted = cv2.bitwise_not(binary_roi)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(inverted, config=config).strip()

def main():
    input_dir = os.path.join("test_dataset", "images")
    out_dir = "presentation_slides"
    os.makedirs(out_dir, exist_ok=True)

    # Automatically grab every single photo in the folder
    valid_exts = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}")
        return

    print(f"\n[INFO] Found {len(image_paths)} images. Building presentation slides...")
    print("-" * 50)

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f" -> Creating slide for {filename}...")

        # 1. Load Image
        img = cv2.imread(img_path)
        if img is None: continue

        # 2. Run Pipeline
        resized, gray, blurred, scale = preprocess(img)
        pred, edges, _ = detect_plate(blurred, gray=gray, image_bgr=resized)

        ocr_text = "NOT DETECTED"
        roi_gray = np.zeros((50, 150), dtype=np.uint8)
        final_clean = np.zeros((50, 150), dtype=np.uint8)

        if pred:
            roi_gray, binary, border_cleared, opened, final_clean = process_plate_region(gray, pred)
            ocr_text = run_ocr(final_clean)
            if not ocr_text:
                ocr_text = "READING FAILED"

        # 3. Setup the 16:9 PowerPoint-style Canvas
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor('#f8f9fa')
        plt.subplots_adjust(wspace=0.1, hspace=0.3)
        fig.suptitle(f"Classical ALPR Pipeline: {filename}", fontsize=22, fontweight='bold', color='#212529')

        ax = axes[0, 0]
        ax.set_title("1. Input & Segmentation", fontsize=14, fontweight='bold')
        ax.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        if pred:
            px, py, pw, ph = pred
            ax.add_patch(mpatches.Rectangle((px,py), pw, ph, lw=3, edgecolor="#00ff00", facecolor="none"))

        ax = axes[0, 1]
        ax.set_title("2. Preprocessing (Blur & Gray)", fontsize=14, fontweight='bold')
        ax.imshow(blurred, cmap="gray")
        ax.axis("off")

        ax = axes[0, 2]
        ax.set_title("3. Canny Edge Detection", fontsize=14, fontweight='bold')
        ax.imshow(edges, cmap="magma")
        ax.axis("off")

        ax = axes[1, 0]
        ax.set_title("4. ROI Extraction (Raw Crop)", fontsize=14, fontweight='bold')
        if pred: ax.imshow(roi_gray, cmap="gray")
        ax.axis("off")

        ax = axes[1, 1]
        ax.set_title("5. Otsu's Binarization & Morphology", fontsize=14, fontweight='bold')
        if pred: ax.imshow(final_clean, cmap="gray")
        ax.axis("off")

        ax = axes[1, 2]
        ax.set_title("6. Tesseract OCR Output", fontsize=14, fontweight='bold')
        ax.axis("off")
        ax.add_patch(mpatches.Rectangle((0, 0.2), 1, 0.6, lw=4, edgecolor="#0d6efd", facecolor="#e9ecef", transform=ax.transAxes))
        ax.text(0.5, 0.5, ocr_text, fontsize=32, fontweight='bold', color="#0d6efd", ha="center", va="center", transform=ax.transAxes)

        # 4. Save the Slide
        out_path = os.path.join(out_dir, f"slide_{os.path.splitext(filename)[0]}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print("\n[SUCCESS] DONE! Open the 'presentation_slides' folder to see your images.")

if __name__ == "__main__":
    main()