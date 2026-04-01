"""
segmentation.py - THE 80% PUSH VERSION
--------------------------------------
An ensemble of 6 detection strategies using classical morphology,
edge gradients, and text-density scoring.
"""

import cv2
import numpy as np

# Tightened constraints based on your 43% baseline
MIN_ASPECT_RATIO = 2.0
MAX_ASPECT_RATIO = 5.5
MIN_AREA_RATIO = 0.015
MAX_AREA_RATIO = 0.20
IDEAL_ASPECT = 4.2  # Closer to standard plate dimensions


def _score_candidate(x, y, w, h, img_h, img_w, gray_img) -> float:
    if h <= 0 or w <= 0: return -1.0
    aspect = w / h
    area_ratio = (w * h) / (img_h * img_w)

    # HARD FILTERS: If it's physically impossible, kill it.
    if not (MIN_ASPECT_RATIO < aspect < MAX_ASPECT_RATIO): return -1.0
    if not (MIN_AREA_RATIO < area_ratio < MAX_AREA_RATIO): return -1.0
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h: return -1.0

    score = 0.0
    roi = gray_img[y:y + h, x:x + w]
    if roi.size == 0: return -1.0

    # 1. Aspect Ratio Match (50 pts)
    ar_score = 1.0 - abs(aspect - IDEAL_ASPECT) / IDEAL_ASPECT
    score += max(0, ar_score) * 50.0

    # 2. Vertical Edge Energy (70 pts) - THIS IS THE SECRET SAUCE
    # Characters are mostly vertical strokes. We check for that "vibration".
    scharr_x = cv2.Scharr(roi, cv2.CV_64F, 1, 0)
    scharr_x = np.absolute(scharr_x)
    energy = np.mean(scharr_x)
    score += min(energy, 100) * 0.7

    # 3. Edge Density (50 pts)
    edges = cv2.Canny(roi, 50, 150)
    density = np.count_nonzero(edges) / roi.size
    if density < 0.08: return -1.0  # If it's too smooth, it's a bumper, not a plate.
    score += density * 100.0

    # 4. Contrast/Variance (30 pts)
    score += (np.std(roi.astype(float)) / 128.0) * 30.0

    # 5. Position Bias (20 pts)
    center_y = (y + h / 2) / img_h
    if 0.4 < center_y < 0.90:
        score += 20.0

    return score


def _apply_nms(candidates, overlap_thresh=0.3):
    """Eliminates overlapping boxes and keeps the best one."""
    if not candidates: return []
    candidates.sort(key=lambda x: x[0], reverse=True)
    keep = []
    while candidates:
        best = candidates.pop(0)
        keep.append(best)
        candidates = [c for c in candidates if _iou(best[1:], c[1:]) < overlap_thresh]
    return keep


def _iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    x1, y1 = max(xA, xB), max(yA, yB)
    x2, y2 = min(xA + wA, xB + wB), min(yA + hA, yB + hB)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (wA * hA) + (wB * hB) - inter
    return inter / union if union > 0 else 0


def _get_candidates(contours, img_h, img_w, gray_img):
    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 400: continue
        x, y, w, h = cv2.boundingRect(cnt)
        s = _score_candidate(x, y, w, h, img_h, img_w, gray_img)
        if s > 0: candidates.append((s, x, y, w, h))
    return candidates


# --- DETECTION STRATEGIES ---

def detect_plate_canny(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _get_candidates(cnts, gray.shape[0], gray.shape[1], gray)


def detect_plate_morphgrad(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _get_candidates(cnts, gray.shape[0], gray.shape[1], gray)


def detect_plate_sobel(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    grad_x = np.absolute(grad_x)
    grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, th = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _get_candidates(cnts, gray.shape[0], gray.shape[1], gray)


def detect_plate_tophat(gray):
    """Strategy 5: Detects light objects (plate) on dark backgrounds (car)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, th = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _get_candidates(cnts, gray.shape[0], gray.shape[1], gray)


def detect_plate_blackhat(gray):
    """Strategy 6: Detects dark objects (text) on light backgrounds (plate)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _get_candidates(cnts, gray.shape[0], gray.shape[1], gray)


def detect_plate_color(image_bgr, gray):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, (0, 0, 150), (180, 60, 255)),
                          cv2.inRange(hsv, (15, 70, 100), (40, 255, 255)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _get_candidates(cnts, gray.shape[0], gray.shape[1], gray)


def detect_plate(blurred, gray=None, image_bgr=None, **kwargs):
    if gray is None: gray = blurred

    all_raw_cands = []
    all_raw_cands.extend(detect_plate_canny(gray))
    all_raw_cands.extend(detect_plate_morphgrad(gray))
    all_raw_cands.extend(detect_plate_sobel(gray))
    all_raw_cands.extend(detect_plate_tophat(gray))
    all_raw_cands.extend(detect_plate_blackhat(gray))
    if image_bgr is not None:
        all_raw_cands.extend(detect_plate_color(image_bgr, gray))

    if not all_raw_cands:
        return None, gray, []

    # Filter overlaps and pick the elite results
    refined = _apply_nms(all_raw_cands)
    best = refined[0]

    return (best[1], best[2], best[3], best[4]), gray, [c[1:] for c in refined]