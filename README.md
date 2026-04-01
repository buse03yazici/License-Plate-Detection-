# License Plate Detection
## LEEN350 – Image Processing Project
**Authors:** Bahaa Yusufoğlu (230303921) · Buse Yazıcı (220301020)  
**University:** Istanbul Arel University – Faculty of Engineering

---

## Project Structure

```
license_plate_detection/
├── main.py             ← Entry point – run this
├── data_loader.py      ← Loads images + XML annotations
├── preprocessing.py    ← Resize · Grayscale · Gaussian blur
├── segmentation.py     ← Canny edge detection · Contour filtering
├── morphology.py       ← Erosion · Dilation · Closing (Otsu binarise)
├── evaluation.py       ← Accuracy · Precision · Recall · F1 · IoU
├── requirements.txt
└── README.md
```

---

## 1 – Install Dependencies

Open a terminal (or PyCharm's built-in terminal) inside the project folder:

```bash
pip install -r requirements.txt
```

---

## 2 – Download the Dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection  
2. Click **Download** (you need a free Kaggle account).  
3. Unzip the downloaded archive.  
4. Rename / arrange the folder so the structure looks like:

```
dataset/
    images/
        Cars0.png
        Cars1.png
        ...
    annotations/
        Cars0.xml
        Cars1.xml
        ...
```

Place the `dataset/` folder **inside** `license_plate_detection/`.

---

## 3 – Run the Project

### From the terminal

```bash
# Quick run (50 images, saves 5 pipeline figures)
python main.py

# Full dataset run
python main.py --dataset dataset/ --max_images 433 --save_visuals 10
```

### From PyCharm

1. Open the `license_plate_detection/` folder as a project.  
2. Set the Python interpreter (File → Settings → Project → Python Interpreter).  
3. Open `main.py` and press the green **Run** button (▶).  
4. To pass arguments: Run → Edit Configurations → Script parameters:  
   `--dataset dataset/ --max_images 433`

---

## 4 – Output

All results are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `<name>_configA.png` | 8-panel pipeline figure for Config A |
| `<name>_configB.png` | 8-panel pipeline figure for Config B |
| `metrics_comparison.png` | Bar chart: Config A vs Config B |

A comparison table is also printed to the console:

```
------------------------------------------
Metric         Config A   Config B
------------------------------------------
Accuracy         0.XXXX     0.XXXX
Precision        0.XXXX     0.XXXX
Recall           0.XXXX     0.XXXX
F1               0.XXXX     0.XXXX
Mean Iou         0.XXXX     0.XXXX
------------------------------------------
```

---

## 5 – Configuration Details

| Parameter | Config A | Config B |
|-----------|----------|----------|
| Canny low threshold | 30 | 50 |
| Canny high threshold | 200 | 150 |
| Erosion / Dilation kernel | 3×3 | 7×7 |
| Closing kernel | 5×5 | 7×7 |

---

## References

1. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.  
2. J. Canny, "A Computational Approach to Edge Detection," *IEEE TPAMI*, 1986.  
3. OpenCV Documentation – https://docs.opencv.org  
4. A. MVD, "Car Plate Detection Dataset," Kaggle, 2020.
