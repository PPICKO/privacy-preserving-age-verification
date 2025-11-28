"""
Inference with Augmentation Script
Applies random augmentation to test model robustness during inference.

For thesis: "Privacy-Preserving Age Verification under GDPR"
Author: Priscila PINTO ICKOWICZ

Usage:
    python inference_with_augmentation.py --image path/to/id_image.jpg
    python inference_with_augmentation.py --image test.jpg --rotation 30
"""

import os
import json
import csv
import random
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AugmentedInferenceConfig:
    """Configuration for inference with augmentation"""
    # Model settings
    model_path: str = r"runs/detect/train/weights/best.pt"
    image_size: int = 640
    confidence_threshold: float = 0.5
    
    # Class names (must match training order)
    class_names: List[str] = field(default_factory=lambda: ["DOB", "GivenName", "Photo", "Surname"])
    
    # OCR settings
    ocr_languages: List[str] = field(default_factory=lambda: ["en", "fr", "nl", "de"])
    ocr_use_gpu: bool = True
    
    # Augmentation settings
    max_rotation: float = 25.0  # Max rotation in degrees (+/-)
    contrast_range: Tuple[float, float] = (0.7, 1.3)  # Alpha for contrast
    brightness_range: Tuple[int, int] = (-40, 40)  # Beta for brightness
    apply_augmentation: bool = True
    
    # Output settings
    output_dir: str = "outputs"
    save_augmented_input: bool = True


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

class ImageAugmenter:
    """Apply augmentations to test model robustness"""
    
    def __init__(self, config: AugmentedInferenceConfig):
        self.config = config
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation to image.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Augmented image
        """
        if not self.config.apply_augmentation:
            return image.copy()
        
        h, w = image.shape[:2]
        
        # Random rotation
        angle = random.uniform(
            -self.config.max_rotation, 
            self.config.max_rotation
        )
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Random brightness/contrast adjustment
        alpha = random.uniform(*self.config.contrast_range)
        beta = random.randint(*self.config.brightness_range)
        adjusted = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)
        
        return adjusted
    
    def get_augmentation_params(self) -> Dict:
        """Get the parameters used for augmentation"""
        return {
            "max_rotation": self.config.max_rotation,
            "contrast_range": self.config.contrast_range,
            "brightness_range": self.config.brightness_range
        }


# ============================================================================
# AUGMENTED INFERENCE PIPELINE
# ============================================================================

class AugmentedInference:
    """Inference pipeline with input augmentation"""
    
    def __init__(self, config: AugmentedInferenceConfig = None):
        self.config = config or AugmentedInferenceConfig()
        self.model = None
        self.ocr_reader = None
        self.augmenter = ImageAugmenter(self.config)
    
    def setup(self) -> bool:
        """Initialize model and OCR reader"""
        print("\n" + "=" * 60)
        print("   AUGMENTED INFERENCE SETUP")
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
        try:
            self.ocr_reader = easyocr.Reader(
                self.config.ocr_languages,
                gpu=self.config.ocr_use_gpu
            )
            print(" OCR reader initialized")
        except Exception as e:
            print(f" Failed to initialize OCR: {e}")
            return False
        
        # Augmentation settings
        print("\n  Augmentation settings:")
        print(f"    Max rotation: ±{self.config.max_rotation}°")
        print(f"    Contrast range: {self.config.contrast_range}")
        print(f"    Brightness range: {self.config.brightness_range}")
        
        return True
    
    def process(self, image_path: str) -> Dict:
        """
        Process image with augmentation.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Results dictionary
        """
        print("\n" + "=" * 60)
        print("   PROCESSING WITH AUGMENTATION")
        print("=" * 60)
        print(f"\n  Input: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f" Could not load image")
            return {"error": "Could not load image"}
        
        print(f" Image loaded ({image.shape[1]}x{image.shape[0]})")
        
        # Apply augmentation
        print("\n  Applying augmentation...")
        augmented = self.augmenter.augment(image)
        print(" Augmentation applied")
        
        # Create output folder
        base_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = Path(self.config.output_dir) / f"{base_name}_AUG_{timestamp}"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save augmented input
        if self.config.save_augmented_input:
            aug_path = output_folder / "augmented_input.png"
            cv2.imwrite(str(aug_path), augmented)
            print(f" Saved augmented input: {aug_path}")
        
        # Run detection on augmented image
        print("\n  Running detection on augmented image...")
        results = self.model(
            augmented,
            imgsz=self.config.image_size,
            conf=self.config.confidence_threshold
        )
        
        # Process detections
        ocr_results = {}
        detections = []
        
        print("\n  Detections:")
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = self.config.class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                print(f"    • {label}: {conf:.1%}")
                
                # Crop from augmented image
                crop = augmented[y1:y2, x1:x2]
                
                # Save crop
                crop_path = output_folder / f"{label}_{conf:.2f}.png"
                cv2.imwrite(str(crop_path), crop)
                
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
        annotated = results[0].plot()
        cv2.imwrite(str(output_folder / "annotated.png"), annotated)
        
        # Save JSON results
        json_path = output_folder / "ocr_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, indent=4, ensure_ascii=False)
        
        # Save CSV results
        csv_path = output_folder / "ocr_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "Value"])
            for key, values in ocr_results.items():
                for v in values:
                    writer.writerow([key, v])
        
        # Summary
        print("\n" + "=" * 60)
        print("   RESULTS")
        print("=" * 60)
        print(f"\n  Output folder: {output_folder}")
        print(f"  Detections: {len(detections)}")
        print("\n  OCR Results:")
        for field, values in ocr_results.items():
            for v in values:
                print(f"    • {field}: {v}")
        print("\n" + "=" * 60 + "\n")
        
        return {
            "image_path": image_path,
            "output_folder": str(output_folder),
            "detections": detections,
            "ocr_results": ocr_results,
            "augmentation_applied": self.config.apply_augmentation
        }
    
    def run(self, image_path: str) -> Dict:
        """Run complete pipeline"""
        if not self.setup():
            return {"error": "Setup failed"}
        
        return self.process(image_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Inference with Random Augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_with_augmentation.py --image test/id_card.jpg
  python inference_with_augmentation.py --image photo.jpg --rotation 30
  python inference_with_augmentation.py --image photo.jpg --no-augmentation
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
        help="Output directory"
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=25.0,
        help="Max rotation angle in degrees (default: 25)"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable augmentation (process original image)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentedInferenceConfig(
        model_path=args.model,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        max_rotation=args.rotation,
        apply_augmentation=not args.no_augmentation
    )
    
    # Run pipeline
    pipeline = AugmentedInference(config)
    results = pipeline.run(args.image)
    
    return results


if __name__ == "__main__":
    main()
