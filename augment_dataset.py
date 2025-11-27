import os
import cv2
import glob
import random
import numpy as np

# === CONFIG ===
DATASET_PATH = r"C:\Users\branq\Desktop\thesis"  # dataset root
IMAGE_DIR = os.path.join(DATASET_PATH, "images", "train")
LABEL_DIR = os.path.join(DATASET_PATH, "labels", "train")
OUTPUT_IMAGE_DIR = os.path.join(DATASET_PATH, "images", "train_aug")
OUTPUT_LABEL_DIR = os.path.join(DATASET_PATH, "labels", "train_aug")

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def yolo_to_xywhn(box, w, h):
    """Convert YOLO normalized box to absolute coords"""
    cls, x, y, bw, bh = box
    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h
    return int(cls), [x1, y1, x2, y2]

def xywh_to_yolo(cls, box, w, h):
    """Convert absolute coords back to YOLO normalized"""
    x1, y1, x2, y2 = box
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    x = (x1 + x2) / 2 / w
    y = (y1 + y2) / 2 / h
    return f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"

def random_perspective(img, boxes):
    """Apply random perspective warp and adjust boxes"""
    h, w = img.shape[:2]

    # random distortion points
    margin = 0.1
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = np.float32([
        [random.uniform(-margin, margin) * w, random.uniform(-margin, margin) * h],
        [w + random.uniform(-margin, margin) * w, random.uniform(-margin, margin) * h],
        [w + random.uniform(-margin, margin) * w, h + random.uniform(-margin, margin) * h],
        [random.uniform(-margin, margin) * w, h + random.uniform(-margin, margin) * h],
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    new_boxes = []
    for cls, (x1, y1, x2, y2) in boxes:
        pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        nx1, ny1 = dst[:, 0].min(), dst[:, 1].min()
        nx2, ny2 = dst[:, 0].max(), dst[:, 1].max()
        nx1, ny1, nx2, ny2 = max(0, nx1), max(0, ny1), min(w, nx2), min(h, ny2)
        if nx2 > nx1 and ny2 > ny1:
            new_boxes.append((cls, [nx1, ny1, nx2, ny2]))

    return warped, new_boxes

# === Process all training images ===
image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGE_DIR, "*.png"))

for img_path in image_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")

    if not os.path.exists(label_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        labels = [list(map(float, line.split())) for line in f.readlines()]

    boxes = [yolo_to_xywhn(l, w, h) for l in labels]

    # Generate N augmentations per image
    for i in range(3):  # create 3 augmented versions
        aug_img, aug_boxes = random_perspective(img, boxes)

        aug_name = f"{base}_aug{i}"
        out_img_path = os.path.join(OUTPUT_IMAGE_DIR, aug_name + ".jpg")
        out_label_path = os.path.join(OUTPUT_LABEL_DIR, aug_name + ".txt")

        cv2.imwrite(out_img_path, aug_img)

        with open(out_label_path, "w") as f:
            for cls, box in aug_boxes:
                f.write(xywh_to_yolo(cls, box, w, h) + "\n")

print("Augmentation complete! New images saved in train_aug/")
