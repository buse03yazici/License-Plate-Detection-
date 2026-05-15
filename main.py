"""
main.py  –  LEEN350 License Plate Detection
Authors: Bahaa Yusufoğlu (230303921) · Buse Yazıcı (220301020)
Istanbul Arel University

Usage:
    python main.py
"""

import argparse, os, time
import cv2, numpy as np
import matplotlib
# Force backend to 'Agg' so it NEVER opens a window or pauses
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_loader   import load_dataset
from preprocessing import preprocess
from segmentation  import detect_plate
from morphology    import process_plate_region
from evaluation    import evaluate_single, aggregate_metrics

# Try importing OCR, but don't crash if it's missing
try:
    from ocr_reader import read_plate, ocr_available
except ImportError:
    def read_plate(roi): return ""
    def ocr_available(): return False

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualise(name, resized, edges, closed, pred, gt_scaled, iou, text, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes: ax.set_facecolor("#161b22")
    tc = "#e6edf3"
    ok = iou >= 0.5
    fig.suptitle(
        f"{name} | Plate: '{text or 'NOT READ'}' | IoU: {iou:.4f} | {'CORRECT (Hit)' if ok else 'MISSED'}",
        fontsize=11, fontweight="bold", color="#00ff88" if ok else "#ff6666", y=1.01
    )

    ax = axes[0]
    ax.set_title("(a) Original + Detection", color=tc, fontsize=10)
    ax.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    handles = []
    if gt_scaled:
        gx, gy, gw, gh = gt_scaled
        ax.add_patch(mpatches.Rectangle((gx,gy),gw,gh, lw=2.5, edgecolor="#ff4444", facecolor="none", ls="--"))
        handles.append(mpatches.Patch(color="#ff4444", label="Ground Truth"))
    if pred:
        px, py, pw, ph = pred
        ax.add_patch(mpatches.Rectangle((px,py),pw,ph, lw=2.5, edgecolor="#00ff88", facecolor="none"))
        handles.append(mpatches.Patch(color="#00ff88", label="Detected"))
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=9, facecolor="#0d1117", labelcolor=tc)

    axes[1].set_title("(b) Canny Edge Map", color=tc, fontsize=10)
    axes[1].imshow(edges, cmap="inferno")
    axes[1].axis("off")

    axes[2].set_title("(c) Cleaned Plate Region", color=tc, fontsize=10)
    if closed is not None and closed.size > 4:
        axes[2].imshow(closed, cmap="gray")
    else:
        axes[2].imshow(np.zeros((40,120), np.uint8), cmap="gray")
        axes[2].text(5, 20, "No plate", color="white", fontsize=9)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

def run_system(data):
    lo, hi, ck = 30, 200, 5  # Standardized detection parameters
    n = len(data)
    results = []

    print(f"\n{'='*65}")
    print(f"  SCANNING ALL {n} IMAGES...")
    print(f"{'='*65}")

    t0 = time.time()

    for i, sample in enumerate(data):
        name    = os.path.splitext(os.path.basename(sample["image_path"]))[0]
        image   = sample["image"]
        gt_bbox = sample["bbox"]

        # Preprocess
        resized, gray, enhanced, scale = preprocess(image)

        # Scale GT bbox
        gt_scaled = None
        if gt_bbox:
            gx, gy, gw, gh = gt_bbox
            gt_scaled = (int(gx*scale), int(gy*scale), int(gw*scale), int(gh*scale))

        # Detect
        pred, edges, _ = detect_plate(enhanced, gray=gray, image_bgr=resized,
                                      low_threshold=lo, high_threshold=hi)

        # Morphology
        roi = closed = None
        if pred:
            roi, _, _, _, closed = process_plate_region(gray, pred, 3, ck)

        # Evaluate
        res = evaluate_single(pred, gt_scaled)
        iou = res["iou"]
        results.append(res)

        # Visualise and save silently
        text = "" # Placeholder if OCR is used later
        save_path = os.path.join(OUTPUT_DIR, f"{name}_result.png")
        visualise(name, resized, edges, closed, pred, gt_scaled, iou, text, save_path)

        # Progress bar
        elapsed = time.time() - t0
        eta = (elapsed / (i+1)) * (n - i - 1)
        print(f"\r  ► Processing {i+1}/{n} | ETA: {eta:.0f}s ", end="", flush=True)

    print("\n")
    return aggregate_metrics(results)

def main():
    ap = argparse.ArgumentParser(description="LEEN350 – License Plate Detection")
    # UPDATED THIS LINE SO IT DEFAULTS TO YOUR NEW FOLDER
    ap.add_argument("--dataset", default="test_dataset", help="Path to dataset folder")
    args = ap.parse_args()

    print("\n" + "="*65)
    print("  LEEN350 – License Plate Detection System")
    print("  Bahaa Yusufoglu & Buse Yazici")
    print("="*65 + "\n")

    # Load data
    try:
        data = load_dataset(args.dataset)
    except Exception as e:
        print(f"[ERROR] Could not load dataset from '{args.dataset}': {e}")
        return

    # Run core system
    metrics = run_system(data)

    # Final Summary
    total_images = metrics["tp"] + metrics["fp"] + metrics["fn"] + metrics["tn"]
    acc_rate = (metrics["tp"] / total_images * 100) if total_images > 0 else 0

    print(f"\n{'='*65}")
    print("  FINAL ACCURACY REPORT")
    print(f"{'='*65}")
    print(f"  ► ACCURACY RATE : {acc_rate:.1f}%")
    print(f"  ► Plates Found  : {metrics['tp']} out of {total_images}")
    print(f"  ► Missed/Wrong  : {metrics['fn'] + metrics['fp']}")
    print(f"{'-'*65}")
    print(f"  Precision       : {metrics['precision']*100:5.1f}%")
    print(f"  Recall          : {metrics['recall']*100:5.1f}%")
    print(f"  F1-Score        : {metrics['f1']*100:5.1f}%")

    out = os.path.abspath(OUTPUT_DIR)
    print(f"\n[INFO] All {total_images} images saved to -> {out}\n")

if __name__ == "__main__":
    main()