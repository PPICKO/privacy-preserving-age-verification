"""
Simple Inference Script
Detects identity document fields and extracts text using OCR.

For thesis: "Privacy-Preserving Age Verification under GDPR"
Author: Priscila PINTO ICKOWICZ

Usage:
    python inference_simple.py --image path/to/id_image.jpg
    python inference_simple.py --image test.jpg --model weights/best.pt
"""

import os
import argparse
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO
import easyocr


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfig:
    """Configuration for inference pipeline"""
    # Model settings
    model_path: str = "runs/detect/train/weights/best.pt"
    image_size: int = 640
    confidence_threshold: float = 0.5
    
    # Class names (must match training order)
    class_names: List[str] = field(default_factory=lambda: ["DOB", "GivenName", "Photo", "Surname"])
    
    # OCR settings
    ocr_languages: List[str] = field(default_factory=lambda: ["en", "fr", "nl", "de"])
    ocr_use_gpu: bool = True
    
    # Output settings
    output_dir: str = "outputs"
    save_crops: bool = True
    save_annotated: bool = True


# ============================================================================
# DATE PARSING
# ============================================================================

class DateParser:
    """Parse dates from various ID document formats"""
    
    DATE_FORMATS = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",  # DD/MM/YYYY
        "%m/%d/%Y", "%m-%d-%Y",              # MM/DD/YYYY
        "%Y-%m-%d", "%Y/%m/%d",              # ISO format
        "%d %b %Y", "%d %B %Y",              # 01 Jan 2000
        "%B %d, %Y",                          # January 01, 2000
    ]
    
    @classmethod
    def parse(cls, date_string: str) -> Optional[date]:
        """
        Parse date from various formats commonly found on ID documents.
        
        Args:
            date_string: Raw date string from OCR
        
        Returns:
            date object or None if parsing fails
        """
        # Clean the string
        clean_string = date_string.strip().replace("  ", " ")
        
        for fmt in cls.DATE_FORMATS:
            try:
                return datetime.strptime(clean_string, fmt).date()
            except ValueError:
                continue
        return None
    
    @staticmethod
    def calculate_age(birth_date: date) -> int:
        """Calculate age from birth date"""
        today = date.today()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age
    
    @staticmethod
    def is_adult(birth_date: date, adult_age: int = 18) -> bool:
        """Check if person is an adult (18+ by default)"""
        return DateParser.calculate_age(birth_date) >= adult_age


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class SimpleInference:
    """Simple inference pipeline for ID document detection"""
    
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.ocr_reader = None
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def setup(self) -> bool:
        """Initialize model and OCR reader"""
        print("\n" + "=" * 60)
        print("   SETUP")
        print("=" * 60)
        
        # Load YOLO model
        print(f"\n  Loading model: {self.config.model_path}")
        try:
            self.model = YOLO(self.config.model_path)
            print(" Model loaded successfully")
        except Exception as e:
            print(f" Failed to load model: {e}")
            return False
        
        # Initialize OCR
        print(f"\n  Initializing OCR (languages: {self.config.ocr_languages})...")
        try:
            self.ocr_reader = easyocr.Reader(
                self.config.ocr_languages, 
                gpu=self.config.ocr_use_gpu
            )
            print(" OCR reader initialized")
        except Exception as e:
            print(f" Failed to initialize OCR: {e}")
            return False
        
        return True
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image through the detection and OCR pipeline.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary with detection and OCR results
        """
        print("\n" + "=" * 60)
        print("   PROCESSING IMAGE")
        print("=" * 60)
        print(f"\n  Input: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f" Could not load image: {image_path}")
            return {"error": "Could not load image"}
        
        print(f" Image loaded ({image.shape[1]}x{image.shape[0]})")
        
        # Run detection
        print("\n  Running detection...")
        results = self.model(
            image_path, 
            imgsz=self.config.image_size, 
            conf=self.config.confidence_threshold
        )
        
        # Process detections
        ocr_results = {}
        detections = []
        
        print("\n" + "-" * 60)
        print("   DETECTION RESULTS")
        print("-" * 60)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = self.config.class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                print(f"\n  {label}: {conf:.1%} confidence")
                print(f"    Box: ({x1}, {y1}) to ({x2}, {y2})")
                
                detection = {
                    "label": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                }
                
                # Crop detected region
                crop = image[y1:y2, x1:x2]
                
                # Save crop
                if self.config.save_crops:
                    crop_path = Path(self.config.output_dir) / f"{label}_{conf:.2f}.png"
                    cv2.imwrite(str(crop_path), crop)
                    print(f"    Saved: {crop_path}")
                    detection["crop_path"] = str(crop_path)
                
                # Run OCR on text fields
                if label != "Photo":
                    # Preprocess for better OCR
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(
                        gray, 0, 255, 
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    
                    # Extract text
                    text = self.ocr_reader.readtext(thresh, detail=0)
                    extracted_text = " ".join(text) if text else "[No text detected]"
                    ocr_results[label] = extracted_text
                    detection["ocr_text"] = extracted_text
                    print(f"    OCR: {extracted_text}")
                
                detections.append(detection)
        
        # Save annotated image
        if self.config.save_annotated:
            annotated = results[0].plot()
            annotated_path = Path(self.config.output_dir) / "annotated.png"
            cv2.imwrite(str(annotated_path), annotated)
            print(f"\n Annotated image saved: {annotated_path}")
        
        return {
            "image_path": image_path,
            "detections": detections,
            "ocr_results": ocr_results
        }
    
    def verify_age(self, ocr_results: Dict) -> Dict:
        """
        Verify age from OCR results.
        
        Args:
            ocr_results: Dictionary with OCR extracted text
        
        Returns:
            Age verification result
        """
        print("\n" + "=" * 60)
        print("   AGE VERIFICATION")
        print("=" * 60)
        
        if "DOB" not in ocr_results:
            print("\n No DOB field detected. Cannot verify age.")
            return {"verified": False, "reason": "No DOB detected"}
        
        dob_text = ocr_results["DOB"]
        birth_date = DateParser.parse(dob_text)
        
        if birth_date is None:
            print(f"\n Could not parse date from: '{dob_text}'")
            return {"verified": False, "reason": f"Could not parse date: {dob_text}"}
        
        age = DateParser.calculate_age(birth_date)
        is_adult = DateParser.is_adult(birth_date)
        
        print(f"\n  Date of Birth: {birth_date}")
        print(f"  Calculated Age: {age} years")
        print(f"  Status: {'ADULT (18+)' if is_adult else 'MINOR'}")
        
        return {
            "verified": True,
            "birth_date": str(birth_date),
            "age_years": age,
            "is_adult": is_adult,
            "status": "adult" if is_adult else "minor"
        }
    
    def run(self, image_path: str) -> Dict:
        """
        Run complete inference pipeline.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Complete results dictionary
        """
        # Setup
        if not self.setup():
            return {"error": "Setup failed"}
        
        # Process image
        results = self.process_image(image_path)
        if "error" in results:
            return results
        
        # Verify age
        age_result = self.verify_age(results.get("ocr_results", {}))
        results["age_verification"] = age_result
        
        # Summary
        print("\n" + "=" * 60)
        print("   SUMMARY")
        print("=" * 60)
        print(f"\n  Detections: {len(results['detections'])}")
        for label, text in results.get("ocr_results", {}).items():
            print(f"    {label}: {text}")
        
        if age_result.get("verified"):
            status = "✓ ADULT" if age_result["is_adult"] else "✗ MINOR"
            print(f"\n  Age Status: {status} ({age_result['age_years']} years)")
        
        print("\n" + "=" * 60 + "\n")
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ID Document Field Detection and OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_simple.py --image test/belgium-id-card.jpg
  python inference_simple.py --image photo.jpg --model weights/best.pt
  python inference_simple.py --image photo.jpg --confidence 0.6
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="test/belgium-id-card.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = InferenceConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
        output_dir=args.output
    )
    
    # Run inference
    pipeline = SimpleInference(config)
    results = pipeline.run(args.image)
    
    return results


if __name__ == "__main__":
    main()
