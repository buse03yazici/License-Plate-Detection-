import cv2
import numpy as np
import os
import shutil


def traditional_plate_detection(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    H, W, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)

    # Kenar Bulma
    scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharr_x = np.absolute(scharr_x)
    scharr_x = np.uint8(255 * scharr_x / np.max(scharr_x))

    # Binarizasyon
    _, thresh = cv2.threshold(scharr_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfoloji: Sadece harf boşluklarını kapatacak kadar
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 4))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    detected_any = False

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = w * h

        # OTOPARK KURALI: Sadece ekranın orta kısımlarına odaklan (Gökyüzü ve yer elenir)
        if y < (H * 0.25) or y > (H * 0.85):
            continue

        # ORAN KURALI: Plaka standartları (2.0 ile 5.5 arası)
        if 2.0 <= aspect_ratio <= 5.5 and 1000 <= area <= (H * W * 0.15):

            roi_bin = thresh[y:y + h, x:x + w]
            density = cv2.countNonZero(roi_bin) / float(area)

            # YOĞUNLUK KURALI: Izgaralar çok boştur (<0.25), Farlar çok doludur (>0.60)
            if 0.25 < density < 0.60:
                detected_any = True
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(output_image, "PLAKA", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output_image if detected_any else None


def process_dataset(folder_path):
    output_dir = "output_results"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    total, detected = 0, 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total += 1
                img_path = os.path.join(root, file)
                result_image = traditional_plate_detection(img_path)

                if result_image is not None:
                    detected += 1
                    cv2.imwrite(os.path.join(output_dir, file), result_image)

    print(f"\n--- FİNAL SONUÇ ---")
    print(f"Toplam Resim: {total}")
    print(f"Tespit Edilen: {detected}")


if __name__ == "__main__":
    process_dataset("clean_dataset")