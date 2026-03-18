import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dosya yolunu senin klasör yapına göre güncelledim
image_path = "dataset/dataset/images/Cars0.png"
img = cv2.imread(image_path)
img = cv2.resize(img, (600, 400))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(blurred, 30, 200)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

result_img = img.copy()
plate_img = None  # Kırpılan plakayı tutacak değişken

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / float(h)

    if 500 < (w * h) < 35000 and 1.5 < aspect_ratio < 7.0:
        if len(approx) == 4:
            cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 3)

            # --- YENİ KISIM: PLAKAYI KIRPMA (CROP) ---
            # Orijinal resimden (img) plaka koordinatlarını kesiyoruz
            plate_img = img[y:y + h, x:x + w]
            break

# Sonuçları yan yana gösterelim
if plate_img is not None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Tespit Edilen Yer")

    axes[1].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Kırpılan Plaka (Segmentasyon Sonucu)")
    plt.show()
else:
    print("Plaka bulunamadı!")