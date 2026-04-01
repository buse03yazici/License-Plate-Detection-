"""
make_xml.py - The Mass Annotator
--------------------------------
Rapidly draw bounding boxes for an entire folder of images.
Automatically skips images that already have an XML file.
"""

import cv2
import os
import glob

# Set your dataset folder here
DATASET_DIR = "test_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNO_DIR = os.path.join(DATASET_DIR, "annotations")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNO_DIR, exist_ok=True)


def run_mass_annotator():
    # Grab all JPG and PNG files in the images folder
    valid_exts = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext.upper())))

    if not image_paths:
        print(f"[ERROR] No images found in '{IMAGES_DIR}'. Add your photos there!")
        return

    print(f"\n[INFO] Found {len(image_paths)} images.")
    print("-" * 50)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(ANNO_DIR, xml_name)

        # SMART SKIP: If XML already exists, immediately skip to the next image
        if os.path.exists(xml_path):
            print(f"[SKIPPING] '{img_name}' already annotated.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read {img_name}")
            continue

        # Scale down 4K images for your monitor
        orig_h, orig_w, _ = img.shape
        scale = 1.0
        MAX_H = 800
        if orig_h > MAX_H:
            scale = MAX_H / orig_h
            display_img = cv2.resize(img, (int(orig_w * scale), MAX_H))
        else:
            display_img = img.copy()

        while True:
            print(f"\nCURRENT IMAGE: {img_name}")
            print(" -> Draw a box and press ENTER or SPACE to save.")
            print(" -> Press 'c' to cancel the box if you messed up.")

            # Open drawing window
            bbox = cv2.selectROI("Mass Annotator (ENTER=Save, C=Cancel Box)", display_img, fromCenter=False,
                                 showCrosshair=True)
            cv2.destroyAllWindows()

            x_disp, y_disp, w_disp, h_disp = bbox

            # If a box was successfully drawn (width and height > 0)
            if w_disp > 0 and h_disp > 0:
                # Scale coordinates back up to original image size
                x = int(x_disp / scale)
                y = int(y_disp / scale)
                w = int(w_disp / scale)
                h = int(h_disp / scale)

                xml_content = f"""<annotation>
    <filename>{img_name}</filename>
    <size>
        <width>{orig_w}</width>
        <height>{orig_h}</height>
        <depth>3</depth>
    </size>
    <object>
        <name>plate</name>
        <bndbox>
            <xmin>{x}</xmin>
            <ymin>{y}</ymin>
            <xmax>{x + w}</xmax>
            <ymax>{y + h}</ymax>
        </bndbox>
    </object>
</annotation>"""
                with open(xml_path, "w") as f:
                    f.write(xml_content)
                print(f"[SAVED] Ground Truth created -> {xml_name}")
                break  # Move to next image

            # If no box was drawn (e.g., they pressed 'c')
            else:
                action = input("[WARNING] No box drawn. [s]kip image, [r]etry drawing, or [q]uit? ").strip().lower()
                if action == 'q':
                    print("Progress saved. Exiting tool...")
                    return
                elif action == 's':
                    print(f"Skipping {img_name}...")
                    break  # Break the while loop, moves to next image
                # If 'r' or anything else, loop repeats and reopens the image

    print("\n[FINISHED] You have reached the end of the folder!")


if __name__ == "__main__":
    run_mass_annotator()