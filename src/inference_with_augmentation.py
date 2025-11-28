import os
import cv2
import json
import csv
import random
from ultralytics import YOLO
import easyocr
from datetime import datetime

# === CONFIG ===
MODEL_PATH = r"runs/detect/train5/weights/best.pt"   # trained YOLO model
IMAGE_PATH = r"C:\Users\branq\Desktop\thesis\test\Belgium-ID-Card-004.jpg"
OUTPUT_DIR = r"outputs"

CLASS_NAMES = ["DOB", "GivenName", "Photo", "Surname"]

# === LOAD YOLO MODEL ===
model = YOLO(MODEL_PATH)

# Initialize OCR (multi-language)
reader = easyocr.Reader(["en", "fr", "nl", "de"])

# === Function: apply random augmentation ===
def augment_image(image):
    h, w = image.shape[:2]

    # Random rotation between -25 and +25 degrees
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Random brightness/contrast adjustment
    alpha = random.uniform(0.7, 1.3)  # contrast
    beta = random.randint(-40, 40)    # brightness
    adjusted = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)

    return adjusted

# === Load and augment ===
image = cv2.imread(IMAGE_PATH)
augmented = augment_image(image)

# Create unique output folder
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
test_output_dir = os.path.join(OUTPUT_DIR, f"{base_name}_AUG_{timestamp}")
os.makedirs(test_output_dir, exist_ok=True)

# Save augmented version for inspection
cv2.imwrite(os.path.join(test_output_dir, "augmented_input.png"), augmented)

# === Run inference on augmented image ===
results = model(augmented, imgsz=640, conf=0.5)

# Store OCR outputs
ocr_results = {}

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = CLASS_NAMES[cls_id]

        # Bounding box coords
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = augmented[y1:y2, x1:x2]

        # Save crop
        crop_filename = os.path.join(test_output_dir, f"{label}_{conf:.2f}.png")
        cv2.imwrite(crop_filename, crop)

        # OCR for text fields
        if label != "Photo":
            text = reader.readtext(crop, detail=0)
            value = " ".join(text) if text else "[No text detected]"
            if label not in ocr_results:
                ocr_results[label] = []
            ocr_results[label].append(value)
            print(f"OCR {label}: {value}")

        print(f"Saved {label} crop -> {crop_filename}")

# Save annotated image
annotated = results[0].plot()
cv2.imwrite(os.path.join(test_output_dir, "annotated.png"), annotated)

# Save JSON
with open(os.path.join(test_output_dir, "ocr_results.json"), "w", encoding="utf-8") as f:
    json.dump(ocr_results, f, indent=4, ensure_ascii=False)

# Save CSV
with open(os.path.join(test_output_dir, "ocr_results.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Field", "Value"])
    for key, values in ocr_results.items():
        for v in values:
            writer.writerow([key, v])

print(f"\nAll results (with augmentation) saved in {test_output_dir}")
