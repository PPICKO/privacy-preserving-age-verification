"""
Dataset Augmentation Script
Applies perspective transformations to training images for data augmentation.

For thesis: "Privacy-Preserving Age Verification under GDPR"
Author: Priscila PINTO ICKOWICZ

Usage:
    python augment_dataset.py
    python augment_dataset.py --input path/to/images --output path/to/augmented
    python augment_dataset.py --num-augmentations 5
"""

import os
import glob
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AugmentationConfig:
    """Configuration for dataset augmentation"""
    # Input paths
    dataset_path: str = r"C:\Users\branq\Desktop\thesis"
    image_subdir: str = "images/train"
    label_subdir: str = "labels/train"
    
    # Output paths
    output_image_subdir: str = "images/train_aug"
    output_label_subdir: str = "labels/train_aug"
    
    # Augmentation settings
    num_augmentations: int = 3  # Number of augmented versions per image
    perspective_margin: float = 0.1  # Max perspective distortion (0.0-0.2)
    
    # File settings
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
    
    @property
    def image_dir(self) -> Path:
        return Path(self.dataset_path) / self.image_subdir
    
    @property
    def label_dir(self) -> Path:
        return Path(self.dataset_path) / self.label_subdir
    
    @property
    def output_image_dir(self) -> Path:
        return Path(self.dataset_path) / self.output_image_subdir
    
    @property
    def output_label_dir(self) -> Path:
        return Path(self.dataset_path) / self.output_label_subdir


# ============================================================================
# AUGMENTATION FUNCTIONS
# ============================================================================

class DatasetAugmenter:
    """Handles dataset augmentation with perspective transforms"""
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self._setup_directories()
    
    def _setup_directories(self):
        """Create output directories if they don't exist"""
        self.config.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.config.output_label_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def yolo_to_absolute(box: List[float], img_width: int, img_height: int) -> Tuple[int, List[float]]:
        """
        Convert YOLO normalized box to absolute coordinates.
        
        Args:
            box: [class_id, x_center, y_center, width, height] (normalized)
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            Tuple of (class_id, [x1, y1, x2, y2]) in absolute coordinates
        """
        cls, x, y, bw, bh = box
        x1 = (x - bw / 2) * img_width
        y1 = (y - bh / 2) * img_height
        x2 = (x + bw / 2) * img_width
        y2 = (y + bh / 2) * img_height
        return int(cls), [x1, y1, x2, y2]
    
    @staticmethod
    def absolute_to_yolo(cls: int, box: List[float], img_width: int, img_height: int) -> str:
        """
        Convert absolute coordinates back to YOLO normalized format.
        
        Args:
            cls: Class ID
            box: [x1, y1, x2, y2] in absolute coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            YOLO format string: "class x_center y_center width height"
        """
        x1, y1, x2, y2 = box
        bw = (x2 - x1) / img_width
        bh = (y2 - y1) / img_height
        x = (x1 + x2) / 2 / img_width
        y = (y1 + y2) / 2 / img_height
        return f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"
    
    def apply_perspective_transform(
        self, 
        image: np.ndarray, 
        boxes: List[Tuple[int, List[float]]]
    ) -> Tuple[np.ndarray, List[Tuple[int, List[float]]]]:
        """
        Apply random perspective transformation to image and adjust bounding boxes.
        
        Args:
            image: Input image (BGR)
            boxes: List of (class_id, [x1, y1, x2, y2]) tuples
        
        Returns:
            Tuple of (transformed_image, transformed_boxes)
        """
        h, w = image.shape[:2]
        margin = self.config.perspective_margin
        
        # Source points (image corners)
        pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Destination points (randomly distorted corners)
        pts_dst = np.float32([
            [random.uniform(-margin, margin) * w, 
             random.uniform(-margin, margin) * h],
            [w + random.uniform(-margin, margin) * w, 
             random.uniform(-margin, margin) * h],
            [w + random.uniform(-margin, margin) * w, 
             h + random.uniform(-margin, margin) * h],
            [random.uniform(-margin, margin) * w, 
             h + random.uniform(-margin, margin) * h],
        ])
        
        # Compute transformation matrix
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        
        # Warp image
        warped = cv2.warpPerspective(
            image, M, (w, h), 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Transform bounding boxes
        transformed_boxes = []
        for cls, (x1, y1, x2, y2) in boxes:
            # Transform box corners
            pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            
            # Get new bounding box from transformed corners
            nx1, ny1 = dst[:, 0].min(), dst[:, 1].min()
            nx2, ny2 = dst[:, 0].max(), dst[:, 1].max()
            
            # Clip to image bounds
            nx1 = max(0, nx1)
            ny1 = max(0, ny1)
            nx2 = min(w, nx2)
            ny2 = min(h, ny2)
            
            # Only keep valid boxes
            if nx2 > nx1 and ny2 > ny1:
                transformed_boxes.append((cls, [nx1, ny1, nx2, ny2]))
        
        return warped, transformed_boxes
    
    def process_image(self, image_path: Path) -> int:
        """
        Process a single image and create augmented versions.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Number of augmented images created
        """
        # Get corresponding label file
        label_path = self.config.label_dir / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            return 0
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   Could not load image: {image_path}")
            return 0
        
        h, w = image.shape[:2]
        
        # Load labels
        with open(label_path, "r") as f:
            labels = [list(map(float, line.split())) for line in f.readlines()]
        
        # Convert to absolute coordinates
        boxes = [self.yolo_to_absolute(label, w, h) for label in labels]
        
        # Generate augmented versions
        count = 0
        for i in range(self.config.num_augmentations):
            # Apply transformation
            aug_image, aug_boxes = self.apply_perspective_transform(image, boxes)
            
            # Generate output paths
            aug_name = f"{image_path.stem}_aug{i}"
            out_image_path = self.config.output_image_dir / f"{aug_name}.jpg"
            out_label_path = self.config.output_label_dir / f"{aug_name}.txt"
            
            # Save augmented image
            cv2.imwrite(str(out_image_path), aug_image)
            
            # Save augmented labels
            with open(out_label_path, "w") as f:
                for cls, box in aug_boxes:
                    f.write(self.absolute_to_yolo(cls, box, w, h) + "\n")
            
            count += 1
        
        return count
    
    def run(self) -> dict:
        """
        Run augmentation on all training images.
        
        Returns:
            Dictionary with statistics
        """
        print("\n" + "=" * 60)
        print("   DATASET AUGMENTATION")
        print("=" * 60)
        
        print(f"\n  Input images:  {self.config.image_dir}")
        print(f"  Input labels:  {self.config.label_dir}")
        print(f"  Output images: {self.config.output_image_dir}")
        print(f"  Output labels: {self.config.output_label_dir}")
        print(f"  Augmentations per image: {self.config.num_augmentations}")
        
        # Get all image files
        image_paths = []
        for ext in self.config.image_extensions:
            image_paths.extend(self.config.image_dir.glob(f"*{ext}"))
            image_paths.extend(self.config.image_dir.glob(f"*{ext.upper()}"))
        
        print(f"\n  Found {len(image_paths)} images to process...")
        
        # Process each image
        total_augmented = 0
        processed = 0
        
        for i, image_path in enumerate(image_paths):
            count = self.process_image(image_path)
            total_augmented += count
            if count > 0:
                processed += 1
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(image_paths)} images...")
        
        # Summary
        print("\n" + "=" * 60)
        print("   AUGMENTATION COMPLETE")
        print("=" * 60)
        print(f"\n   Images processed: {processed}")
        print(f"   Augmented images created: {total_augmented}")
        print(f"   Output directory: {self.config.output_image_dir}")
        print("\n" + "=" * 60 + "\n")
        
        return {
            "images_processed": processed,
            "augmented_created": total_augmented,
            "output_dir": str(self.config.output_image_dir)
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Dataset augmentation with perspective transforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augment_dataset.py
  python augment_dataset.py --num-augmentations 5
  python augment_dataset.py --dataset-path /path/to/dataset
        """
    )
    
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default=r"C:\Users\branq\Desktop\thesis",
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--num-augmentations", 
        type=int, 
        default=3,
        help="Number of augmented versions per image (default: 3)"
    )
    parser.add_argument(
        "--margin", 
        type=float, 
        default=0.1,
        help="Perspective distortion margin 0.0-0.2 (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = AugmentationConfig(
        dataset_path=args.dataset_path,
        num_augmentations=args.num_augmentations,
        perspective_margin=args.margin
    )
    
    # Run augmentation
    augmenter = DatasetAugmenter(config)
    augmenter.run()


if __name__ == "__main__":
    main()
