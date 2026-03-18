import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "dataset/dataset/images/Cars0.png"
img = cv2.imread(image_path)
img = cv2.resize(img, (600, 400))

# 1. Pre-processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(blurred, 30, 200)

# 2. Find Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

result_img = img.copy()

# 3. Geometric Filtering
for c in contours:
    # Calculate the perimeter of the contour
    peri = cv2.arcLength(c, True)
    # Approximate the shape to a polygon
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(c)
    area = w * h

    if h == 0: continue
    aspect_ratio = w / float(h)

    # Filter 1: Check if the shape size proportions make sense for a plate
    if 500 < area < 35000 and 1.5 < aspect_ratio < 7.0:
        # Filter 2: Check if the approximated polygon has exactly 4 corners (a rectangle)
        if len(approx) == 4:
            # Draw a green contour exactly around the 4 corners
            cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 3)
            break  # Plate found, exit the loop

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title("Final Detection using Geometric Approximation")
plt.axis("off")
plt.show()