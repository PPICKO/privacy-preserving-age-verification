# privacy-preserving-age-verification
Privacy-Preserving Age Verification System using YOLOv8 and JWT Tokens
# Privacy-Preserving Age Verification System

A GDPR-compliant age verification system using YOLOv8 for identity document field detection and JWT-based encrypted tokens for privacy-preserving credential exchange.

## Overview

This project implements a privacy-by-design age verification system that:
- Detects key fields (Date of Birth, Name, Photo) on identity documents using YOLOv8
- Extracts text using EasyOCR with multi-language support
- Generates cryptographically signed JWT tokens containing only boolean age status
- Binds tokens to specific devices to prevent credential sharing
- Deletes all personal data immediately after verification

## Features

- **High Accuracy Detection**: 98.9% mAP@0.5 on identity document field detection
- **Multi-Language OCR**: Supports English, French, Dutch, and German text extraction
- **Privacy-Preserving**: No personal data stored; only boolean "is adult" status retained
- **Fast Processing**: End-to-end verification in under 700ms
- **Device Binding**: Tokens bound to device fingerprints to prevent sharing

## Model Performance

| Metric | Value |
|--------|-------|
| Overall mAP@0.5 | 98.9% |
| Overall mAP@0.5:0.95 | 74.0% |
| Precision | 98.0% |
| Recall | 95.9% |
| F1-Score | 0.97 |

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| Photo | 0.98 | 0.95 | 98.7% | 69.2% |
| DOB | 0.97 | 0.96 | 98.8% | 68.3% |
| GivenName | 0.99 | 0.99 | 99.5% | 90.9% |
| Surname | 0.97 | 0.93 | 98.7% | 67.8% |

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PPICKO/privacy-preserving-age-verification.git
cd privacy-preserving-age-verification
```

2. Create a virtual environment:
```bash
conda create -n age-verify python=3.12
conda activate age-verify
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For GPU support (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Running Inference

```bash
python inference_simple.py
```

Or use the YOLO CLI:
```bash
yolo predict model=runs/detect/train/weights/best.pt source="path/to/id_image.jpg" imgsz=640 conf=0.5
```

### Training Your Own Model

1. Prepare your dataset in YOLO format:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── idcard.yaml
```

2. Create the dataset configuration file (`idcard.yaml`):
```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: DOB
  1: GivenName
  2: Photo
  3: Surname
```

3. Train the model:
```bash
yolo train model=yolov8s.pt data=idcard.yaml epochs=50 imgsz=640 batch=16
```

## Project Structure

```
├── README.md
├── requirements.txt
├── idcard.yaml                 # Dataset configuration
├── inference_simple.py         # Inference script with OCR
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt     # Trained model weights
├── images/                     # Training images
│   ├── train/
│   └── val/
├── labels/                     # YOLO format annotations
│   ├── train/
│   └── val/
└── outputs/                    # Inference results
```

## Training Details

- **Model**: YOLOv8s (small)
- **Epochs**: 50
- **Image Size**: 640×640
- **Batch Size**: 16
- **Optimizer**: AdamW (lr=0.00125)
- **Training Time**: ~27 minutes on NVIDIA RTX A1000
- **Dataset**: 562 training images, 212 validation images

## Hardware Requirements

### Development Environment
- CPU: 12th Gen Intel Core i7-12700H
- RAM: 16GB DDR5
- GPU: NVIDIA RTX A1000 (4GB VRAM)
- Storage: 512GB NVMe SSD

### Minimum Requirements
- CPU: Any modern multi-core processor
- RAM: 8GB
- GPU: Optional (CPU inference supported)
- Storage: 2GB free space

## License

This project is part of a Master's thesis at Université Libre de Bruxelles (ULB).

## Citation

If you use this work, please cite:

```
@mastersthesis{pintoickowicz,
  author  = {Pinto Ickowicz, Priscila},
  title   = {Privacy-Preserving Age Verification under GDPR: AI-Based Identity Extraction and Encrypted Token Exchange},
  school  = {Université Libre de Bruxelles},
  year    = {2025},
  type    = {Specialized Master's Thesis}
}
```

## Acknowledgments

- Supervisors: Prof. Dimitris Sacharidis, Prof. Jan Tobias Mühlberg
- Faculty of Science, Université Libre de Bruxelles
- Ultralytics for YOLOv8
- EasyOCR for text extraction

## Contact

For questions or collaboration, please open an issue on this repository.
