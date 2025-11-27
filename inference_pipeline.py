import os
import cv2
import json
import csv
from ultralytics import YOLO
import easyocr
from datetime import datetime

# === CONFIG ===
MODEL_PATH = r"runs/detect/train5/weights/best.pt"   # trained YOLO model
IMAGE_PATH = r"C:\Users\branq\Desktop\thesis\test\Belgium-ID-Card-004.jpg"  # input image
OUTPUT_DIR = r"outputs"   # main save folder

# Class names (must match classes.txt order!)
CLASS_NAMES = ["DOB", "GivenName", "Photo", "Surname"]

# === LOAD YOLO MODEL ===
model = YOLO(MODEL_PATH)

# Run inference
results = model(IMAGE_PATH, imgsz=640, conf=0.5)

# Load image for cropping
image = cv2.imread(IMAGE_PATH)
h, w, _ = image.shape

# Create unique folder per test (timestamp or filename-based)
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
test_output_dir = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}")
os.makedirs(test_output_dir, exist_ok=True)

# Initialize OCR (multi-language: English, French, Dutch, German)
reader = easyocr.Reader(["en", "fr", "nl", "de"])

# Store OCR outputs
ocr_results = {}

# Process detections
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = CLASS_NAMES[cls_id]

        # Bounding box coords
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Crop region
        crop = image[y1:y2, x1:x2]

        # Save crop
        crop_filename = os.path.join(test_output_dir, f"{label}_{conf:.2f}.png")
        cv2.imwrite(crop_filename, crop)

        # OCR only for text fields
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

# === SAVE RESULTS TO JSON ===
json_path = os.path.join(test_output_dir, "ocr_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(ocr_results, f, indent=4, ensure_ascii=False)

# === SAVE RESULTS TO CSV ===
csv_path = os.path.join(test_output_dir, "ocr_results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Field", "Value"])
    for key, values in ocr_results.items():
        for v in values:
            writer.writerow([key, v])

print(f"\nAll results saved in {test_output_dir}")
