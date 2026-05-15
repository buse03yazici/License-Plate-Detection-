"""
viewer.py – The Complete ALPR Pipeline
LEEN350 Project: Abdulrahman & Bahaa
------------------------------------------
Final Stage: Detection + Morphology + OCR
"""

import argparse
import os
import cv2
import numpy as np
import pytesseract

from data_loader   import load_dataset
from preprocessing import preprocess
from segmentation  import detect_plate
from morphology    import process_plate_region
from evaluation    import evaluate_single

# --- OCR CONFIGURATION ---
# Point this to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

OUTPUT_DIR = "output_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_detections(image_bgr, pred, gt_bbox, scale, name, iou, ocr_text):
    vis = image_bgr.copy()

    # Header Overlay
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (vis.shape[1], 80), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

    # Info Text
    cv2.putText(vis, f"SYSTEM: {name} | IoU: {iou:.2f}", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # OCR Text Display (Big and Bold if found)
    if ocr_text and ocr_text != "N/A":
        cv2.putText(vis, f"DETECTED PLATE: {ocr_text}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

    if gt_bbox:
        gx, gy, gw, gh = gt_bbox
        cv2.rectangle(vis, (int(gx*scale), int(gy*scale)),
                      (int((gx+gw)*scale), int((gy+gh)*scale)), (0, 0, 255), 2)

    if pred:
        px, py, pw, ph = pred
        cv2.rectangle(vis, (px, py), (px+pw, py+ph), (0, 255, 0), 2)

    return vis

def build_pipeline_strip(gray, blurred, edges, binary_cleaned, final_roi):
    """Stitches the steps into a horizontal UI strip."""
    H = 150
    def panel(img, label):
        if img is None or img.size == 0:
            img = np.zeros((H, H*2, 3), np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h_o, w_o = img.shape[:2]
        img_res = cv2.resize(img, (int(H * (w_o/h_o)), H))
        bar = np.zeros((25, img_res.shape[1], 3), np.uint8)
        cv2.putText(bar, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return np.vstack([bar, img_res])

    steps = [
        panel(gray, "1. Gray"),
        panel(blurred, "2. Prepped"),
        panel(edges, "3. Canny/Ensemble"),
        panel(binary_cleaned, "4. Binary Text"),
        panel(final_roi, "5. Final Crop")
    ]

    full_strip = steps[0]
    for i in range(1, len(steps)):
        full_strip = np.hstack([full_strip, steps[i]])
    return full_strip

def run_ocr(binary_roi):
    """Runs Tesseract on the cleaned morphological ROI."""
    if binary_roi is None or binary_roi.size == 0: return "N/A"

    # Tesseract loves Black text on White background
    # Our morphology is white-on-black, so we invert it
    inverted = cv2.bitwise_not(binary_roi)

    # PSM 7: Treat the image as a single text line
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(inverted, config=config)
    return text.strip()

def run_viewer(dataset_dir, max_images=30):
    data = load_dataset(dataset_dir)[:max_images]
    total = len(data)
    WIN = "LEEN350 Master Pipeline"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    for idx, sample in enumerate(data):
        img = sample["image"]
        name = os.path.splitext(os.path.basename(sample["image_path"]))[0]

        # 1. Pipeline execution
        resized, gray, blurred, scale = preprocess(img)
        pred, edges, _ = detect_plate(blurred, gray=gray, image_bgr=resized)

        ocr_result = "N/A"
        final_roi, binary_cleaned = None, None

        if pred:
            # 2. Morphological refinement
            roi_gray, binary, eroded, dilated, closed = process_plate_region(gray, pred)
            final_roi = roi_gray
            binary_cleaned = closed

            # 3. OCR (Only if the detection is likely correct)
            gt_scaled = None
            if sample["bbox"]:
                gx, gy, gw, gh = sample["bbox"]
                gt_scaled = (int(gx*scale), int(gy*scale), int(gw*scale), int(gh*scale))

            iou = evaluate_single(pred, gt_scaled)["iou"]
            if iou >= 0.35:
                ocr_result = run_ocr(closed)
        else:
            iou = 0.0

        # 4. Visualization
        det_view = draw_detections(resized, pred, sample["bbox"], scale, name, iou, ocr_result)
        strip = build_pipeline_strip(gray, blurred, edges, binary_cleaned, final_roi)

        # Combine
        max_w = max(det_view.shape[1], strip.shape[1])
        canvas = np.zeros((det_view.shape[0] + strip.shape[0] + 10, max_w, 3), np.uint8)
        canvas[:det_view.shape[0], :det_view.shape[1]] = det_view
        canvas[det_view.shape[0]+10:, :strip.shape[1]] = strip

        cv2.imshow(WIN, canvas)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_final.png"), canvas)

        if (cv2.waitKey(0) & 0xFF) == ord('q'): bيreak

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # UPDATED THIS LINE SO IT DEFAULTS TO YOUR NEW FOLDER
    parser.add_argument("--dataset", type=str, default="test_dataset")
    args = parser.parse_args()
    run_viewer(args.dataset)