"""
Privacy-Preserving Age Verification - Inference Script
Detects identity document fields and extracts text using OCR.

Usage:
    python inference_simple.py --image path/to/id_image.jpg
    python inference_simple.py  # Uses default test image
"""

import os
import cv2
import argparse
from datetime import datetime, date
from ultralytics import YOLO
import easyocr

# Configuration
MODEL_PATH = "runs/detect/train/weights/best.pt"
OUTPUT_DIR = "outputs"
CLASS_NAMES = ["DOB", "GivenName", "Photo", "Surname"]
CONFIDENCE_THRESHOLD = 0.5
OCR_LANGUAGES = ["en", "fr", "nl", "de"]  # English, French, Dutch, German


def parse_date(date_string):
    """
    Parse date from various formats commonly found on ID documents.
    Returns a date object or None if parsing fails.
    """
    date_formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",  # DD/MM/YYYY
        "%m/%d/%Y", "%m-%d-%Y",              # MM/DD/YYYY
        "%Y-%m-%d", "%Y/%m/%d",              # ISO format
        "%d %b %Y", "%d %B %Y",              # 01 Jan 2000
        "%B %d, %Y",                          # January 01, 2000
    ]
    
    # Clean the string
    clean_string = date_string.strip().replace("  ", " ")
    
    for fmt in date_formats:
        try:
            return datetime.strptime(clean_string, fmt).date()
        except ValueError:
            continue
    return None


def calculate_age(birth_date):
    """Calculate age from birth date."""
    today = date.today()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


def is_adult(birth_date, adult_age=18):
    """Check if person is an adult (18+ by default)."""
    return calculate_age(birth_date) >= adult_age


def main(image_path):
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Run inference
    print(f"Processing image: {image_path}")
    results = model(image_path, imgsz=640, conf=CONFIDENCE_THRESHOLD)
    
    # Load image for cropping
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Initialize OCR reader
    print("Initializing OCR reader...")
    reader = easyocr.Reader(OCR_LANGUAGES, gpu=True)
    
    # Process detections
    ocr_results = {}
    print("\n" + "="*50)
    print("DETECTION RESULTS")
    print("="*50)
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = CLASS_NAMES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            print(f"\n{label}: confidence {conf:.2%}")
            print(f"  Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Crop detected region
            crop = image[y1:y2, x1:x2]
            crop_path = os.path.join(OUTPUT_DIR, f"{label}_{conf:.2f}.png")
            cv2.imwrite(crop_path, crop)
            print(f"  Saved crop: {crop_path}")
            
            # Run OCR on text fields (not Photo)
            if label != "Photo":
                # Preprocess for better OCR
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Extract text
                text = reader.readtext(thresh, detail=0)
                extracted_text = " ".join(text) if text else "[No text detected]"
                ocr_results[label] = extracted_text
                print(f"  OCR result: {extracted_text}")
    
    # Save annotated image
    annotated = results[0].plot()
    annotated_path = os.path.join(OUTPUT_DIR, "annotated.png")
    cv2.imwrite(annotated_path, annotated)
    print(f"\nAnnotated image saved: {annotated_path}")
    
    # Process DOB if detected
    print("\n" + "="*50)
    print("AGE VERIFICATION")
    print("="*50)
    
    if "DOB" in ocr_results:
        dob_text = ocr_results["DOB"]
        birth_date = parse_date(dob_text)
        
        if birth_date:
            age = calculate_age(birth_date)
            adult_status = is_adult(birth_date)
            
            print(f"\nDate of Birth: {birth_date}")
            print(f"Calculated Age: {age} years")
            print(f"Adult Status (18+): {'YES - ADULT' if adult_status else 'NO - MINOR'}")
        else:
            print(f"\nCould not parse date from: {dob_text}")
            print("Manual verification required.")
    else:
        print("\nNo DOB field detected. Cannot verify age.")
    
    # Summary
    print("\n" + "="*50)
    print("EXTRACTED INFORMATION")
    print("="*50)
    for field, text in ocr_results.items():
        print(f"  {field}: {text}")
    
    return ocr_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID Document Field Detection and OCR")
    parser.add_argument("--image", "-i", type=str, default="test/belgium-id-card.jpg",
                        help="Path to input image")
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH,
                        help="Path to YOLO model weights")
    args = parser.parse_args()
    
    if args.model != MODEL_PATH:
        MODEL_PATH = args.model
    
    main(args.image)
