"""
Inference Pipeline Script
Full detection pipeline with OCR and structured output (JSON, CSV).

For thesis: "Privacy-Preserving Age Verification under GDPR"
Author: Priscila PINTO ICKOWICZ

Usage:
    python inference_pipeline.py --image path/to/id_image.jpg
    python inference_pipeline.py --image test.jpg --output results/
"""

import os
import json
import csv
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO
import easyocr


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for inference pipeline"""
    # Model settings
    model_path: str = r"runs/detect/train/weights/best.pt"
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
    save_json: bool = True
    save_csv: bool = True
    create_timestamped_folder: bool = True


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """Full inference pipeline with structured output"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.model = None
        self.ocr_reader = None
    
    def setup(self) -> bool:
        """Initialize model and OCR reader"""
        print("\n" + "=" * 60)
        print("   INFERENCE PIPELINE SETUP")
        print("=" * 60)
        
        # Load YOLO model
        print(f"\n  Loading YOLO model: {self.config.model_path}")
        try:
            self.model = YOLO(self.config.model_path)
            print(" Model loaded successfully")
        except Exception as e:
            print(f" Failed to load model: {e}")
            return False
        
        # Initialize OCR
        print(f"\n  Initializing OCR...")
        print(f"    Languages: {', '.join(self.config.ocr_languages)}")
        print(f"    GPU: {'Enabled' if self.config.ocr_use_gpu else 'Disabled'}")
        try:
            self.ocr_reader = easyocr.Reader(
                self.config.ocr_languages,
                gpu=self.config.ocr_use_gpu
            )
            print(" OCR reader initialized")
        except Exception as e:
            print(f" Failed to initialize OCR: {e}")
            return False
        
        print("\n Setup complete!")
        return True
    
    def _create_output_folder(self, image_path: str) -> Path:
        """Create output folder for this run"""
        base_name = Path(image_path).stem
        
        if self.config.create_timestamped_folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{base_name}_{timestamp}"
        else:
            folder_name = base_name
        
        output_path = Path(self.config.output_dir) / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def process(self, image_path: str) -> Dict:
        """
        Process image through complete pipeline.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "=" * 60)
        print("   PROCESSING")
        print("=" * 60)
        print(f"\n  Input image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f" Could not load image: {image_path}")
            return {"error": "Could not load image", "image_path": image_path}
        
        h, w = image.shape[:2]
        print(f" Image loaded ({w}x{h})")
        
        # Create output folder
        output_folder = self._create_output_folder(image_path)
        print(f" Output folder: {output_folder}")
        
        # Run detection
        print("\n  Running YOLOv8 detection...")
        results = self.model(
            image_path,
            imgsz=self.config.image_size,
            conf=self.config.confidence_threshold
        )
        
        # Process detections
        ocr_results = {}
        detections = []
        
        print("\n  Processing detections:")
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = self.config.class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                print(f"    • {label}: {conf:.1%} at ({x1},{y1})-({x2},{y2})")
                
                # Crop region
                crop = image[y1:y2, x1:x2]
                
                # Save crop
                if self.config.save_crops:
                    crop_filename = output_folder / f"{label}_{conf:.2f}.png"
                    cv2.imwrite(str(crop_filename), crop)
                
                # OCR for text fields
                ocr_text = None
                if label != "Photo":
                    text = self.ocr_reader.readtext(crop, detail=0)
                    ocr_text = " ".join(text) if text else "[No text detected]"
                    
                    if label not in ocr_results:
                        ocr_results[label] = []
                    ocr_results[label].append(ocr_text)
                    
                    print(f"      OCR: {ocr_text}")
                
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2],
                    "ocr_text": ocr_text
                })
        
        # Save annotated image
        if self.config.save_annotated:
            annotated = results[0].plot()
            annotated_path = output_folder / "annotated.png"
            cv2.imwrite(str(annotated_path), annotated)
            print(f"\n Saved annotated image")
        
        # Save JSON results
        if self.config.save_json:
            json_path = output_folder / "ocr_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_results, f, indent=4, ensure_ascii=False)
            print(f" Saved JSON results")
        
        # Save CSV results
        if self.config.save_csv:
            csv_path = output_folder / "ocr_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Field", "Value"])
                for key, values in ocr_results.items():
                    for v in values:
                        writer.writerow([key, v])
            print(f" Saved CSV results")
        
        # Summary
        print("\n" + "=" * 60)
        print("   RESULTS SUMMARY")
        print("=" * 60)
        print(f"\n  Total detections: {len(detections)}")
        print(f"  OCR fields extracted: {len(ocr_results)}")
        print(f"  Output folder: {output_folder}")
        
        print("\n  Extracted text:")
        for field, values in ocr_results.items():
            for v in values:
                print(f"    • {field}: {v}")
        
        print("\n" + "=" * 60 + "\n")
        
        return {
            "image_path": image_path,
            "output_folder": str(output_folder),
            "detections": detections,
            "ocr_results": ocr_results
        }
    
    def run(self, image_path: str) -> Dict:
        """
        Run complete pipeline.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Results dictionary
        """
        if not self.setup():
            return {"error": "Setup failed"}
        
        return self.process(image_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ID Document Detection Pipeline with OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_pipeline.py --image test/belgium-id-card.jpg
  python inference_pipeline.py --image photo.jpg --output results/
  python inference_pipeline.py --image photo.jpg --confidence 0.6
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to YOLO model weights"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't create timestamped output folders"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        model_path=args.model,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        create_timestamped_folder=not args.no_timestamp
    )
    
    # Run pipeline
    pipeline = InferencePipeline(config)
    results = pipeline.run(args.image)
    
    return results


if __name__ == "__main__":
    main()
