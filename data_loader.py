"""
data_loader.py
--------------
Loads images and ground truth bounding-box annotations from the
Car Plate Detection dataset (Kaggle – andrewmvd).

Dataset layout expected:
    dataset/
        images/   *.jpg  (or *.png)
        annotations/  *.xml   (PASCAL VOC format)

Each XML file looks like:
    <annotation>
      <object>
        <bndbox>
          <xmin>...</xmin><ymin>...</ymin>
          <xmax>...</xmax><ymax>...</ymax>
        </bndbox>
      </object>
    </annotation>
"""

import os
import glob
import xml.etree.ElementTree as ET
import cv2


def load_dataset(dataset_dir: str):
    """
    Loads all (image, ground_truth_bbox) pairs from dataset_dir.

    Parameters
    ----------
    dataset_dir : str
        Root folder that contains `images/` and `annotations/` sub-folders.

    Returns
    -------
    data : list of dict
        Each dict has keys:
          - 'image_path'  : str
          - 'image'       : np.ndarray  (BGR, original size)
          - 'bbox'        : (x, y, w, h) or None if no annotation found
    """
    images_dir      = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(dataset_dir, "annotations")

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in '{images_dir}'. "
            "Make sure the dataset is downloaded and placed correctly."
        )

    data = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(annotations_dir, stem + ".xml")

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue

        bbox = _parse_annotation(xml_path) if os.path.exists(xml_path) else None

        data.append({
            "image_path": img_path,
            "image":      image,
            "bbox":       bbox,   # (x, y, w, h) in original image coords
        })

    print(f"[INFO] Loaded {len(data)} images from '{dataset_dir}'.")
    return data


def _parse_annotation(xml_path: str):
    """
    Parses a PASCAL VOC XML annotation and returns the first bounding box
    as (x, y, w, h).
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj  = root.find("object")
        if obj is None:
            return None
        bb   = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        return (xmin, ymin, xmax - xmin, ymax - ymin)
    except Exception as e:
        print(f"[WARNING] Could not parse annotation '{xml_path}': {e}")
        return None
