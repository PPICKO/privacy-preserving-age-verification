# Privacy-Preserving Age Verification System

A GDPR-compliant age verification system using AI-based identity document processing and encrypted token exchange.

**Master's Thesis Project** - Université Libre de Bruxelles  
**Author:** Priscila PINTO ICKOWICZ

---

## Overview

This system provides privacy-preserving age verification by:
1. **Detecting** identity document fields using YOLOv8 object detection
2. **Extracting** date of birth via OCR (EasyOCR)
3. **Calculating** age locally without transmitting personal data
4. **Generating** cryptographic tokens that prove age status without revealing the actual birthdate

### Key Features

- **Privacy-First**: No personal data leaves the device
- **GDPR Compliant**: Implements data minimization principles
- **High Accuracy**: 98.9% mAP@0.5 for document field detection
- **Fast Processing**: ~277ms end-to-end verification
- **Secure Tokens**: RSA-2048 signed JWT tokens with device binding

---

## System Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Webcam    │────▶│   YOLOv8    │────▶│   EasyOCR   │────▶│    Token    │
│   Input     │     │  Detection  │     │  Extraction │     │  Generation │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │                    │                    │
                          ▼                    ▼                    ▼
                    DOB, Photo,           Date of Birth        JWT Token
                    Name fields           extracted            (is_adult: bool)
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for faster inference)
- Webcam

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PPICKO/privacy-preserving-age-verification.git
   cd privacy-preserving-age-verification
   ```

2. **Create virtual environment:**
   ```bash
   conda create -n yolov8_env python=3.9
   conda activate yolov8_env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model:**
   - Place `my_model.pt` in the project root directory
   - Or update the `MODEL_PATH` in `id_detection.py`

---

## Usage

### Basic Age Verification

Run the main detection system:

```bash
python id_detection.py
```

**Instructions:**
1. Hold your ID card in front of the webcam
2. Ensure the DOB field is clearly visible
3. The system will automatically capture when confidence thresholds are met
4. Press `q` to quit

**Output:**
- Age status (ADULT/MINOR/UNKNOWN)
- Annotated snapshot saved to `outputs/` folder
- OCR results and parsed date

### Token Generation

After successful age verification:

```python
from token_system import AgeVerificationToken, TokenConfig

config = TokenConfig(VALIDITY_DAYS=30)
token_gen = AgeVerificationToken(config)

# Generate token
result = token_gen.generate_token(is_adult=True, age_years=25)
print(result['token'])

# Validate token
is_valid, payload = token_gen.validate_token(result['token'])
```

---

## Project Structure

```
privacy-preserving-age-verification/
├── id_detection.py          # Main detection pipeline
├── token_system.py          # JWT token generation/validation
├── benchmarks/
│   ├── e2e_benchmark_v2.py  # End-to-end pipeline timing
│   └── token_benchmark.py   # Token security & performance tests
├── models/
│   └── my_model.pt          # Trained YOLOv8 model
├── outputs/                  # Detection results
├── requirements.txt
└── README.md
```

---

## Benchmarks

### Running the Token Benchmark

Tests token validation latency, security attacks, and scalability:

```bash
python benchmarks/token_benchmark.py
```

**Metrics collected:**
- Token validation latency (mean, median, P95, P99)
- Expired token rejection rate
- Security attack resistance (payload modification, signature forgery, algorithm confusion)
- Concurrent validation scalability

**Sample Results:**
| Metric | Result |
|--------|--------|
| Validation latency | ~0.1ms |
| Expired token rejection | 100% |
| Security attack rejection | 100% |

### Running the End-to-End Benchmark

Tests complete pipeline timing from detection to token generation:

```bash
# Using a test image (recommended)
python benchmarks/e2e_benchmark_v2.py --image path/to/id_card.jpg

# Using webcam (interactive)
python benchmarks/e2e_benchmark_v2.py
```

**Metrics collected:**
- YOLOv8 detection latency
- OCR processing latency
- Date parsing latency
- Token generation latency
- Total end-to-end latency

**Sample Results:**
| Stage | Latency | % of Total |
|-------|---------|------------|
| YOLOv8 Detection | 44.6ms | 16.1% |
| OCR Processing | 71.7ms | 25.9% |
| Date Parsing | 0.2ms | 0.1% |
| Token Generation | 159.1ms | 57.4% |
| **Total** | **277ms** | 100% |

---

## Model Performance

### YOLOv8 Detection Results

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| DOB | 97.5% | 97.7% | 99.0% |
| GivenName | 99.0% | 93.3% | 98.2% |
| Photo | 99.3% | 99.3% | 99.5% |
| Surname | 97.5% | 94.0% | 98.7% |
| **Overall** | **98.3%** | **96.1%** | **98.9%** |

### Training Details

- **Model:** YOLOv8s (small variant)
- **Dataset:** 774 images (562 train / 212 val)
- **Epochs:** 100
- **Image Size:** 640x640
- **Hardware:** NVIDIA Tesla T4 GPU

---

## Security Features

### Token Security

- **RSA-2048 Signatures:** Cryptographic proof of authenticity
- **Device Fingerprinting:** Tokens bound to specific devices
- **Expiration:** Configurable validity period (default: 30 days)
- **Algorithm Enforcement:** Strict RS256 validation

### Attack Resistance

| Attack Type | Protection |
|-------------|------------|
| Payload Modification | RSA signature verification |
| Signature Forgery | 2048-bit key strength |
| Algorithm Confusion | Strict algorithm enforcement |
| Token Replay | Expiration + device binding |

---

## Configuration

### Detection Settings (`id_detection.py`)

```python
@dataclass
class Config:
    MODEL_PATH = "my_model.pt"
    THRESHOLDS = {
        "DOB": 0.80,
        "Photo": 0.80,
        "GivenName": 0.80,
        "Surname": 0.80
    }
    IMGSZ = 640
    MAX_RETRY_ATTEMPTS = 5
```

### Token Settings (`token_system.py`)

```python
@dataclass
class TokenConfig:
    VALIDITY_DAYS = 30
    ISSUER = "AgeVerification-Thesis-System"
    KEY_SIZE = 2048
    ENABLE_DEVICE_BINDING = True
```

---

## Requirements

```
ultralytics>=8.0.0
easyocr>=1.7.0
opencv-python>=4.8.0
pyjwt>=2.8.0
cryptography>=41.0.0
qrcode>=7.4.0
psutil>=5.9.0
numpy>=1.24.0
torch>=2.0.0
```

---

## License

This project is part of a Master's thesis at Université Libre de Bruxelles.

---

## Acknowledgments

- **Institution:** Université Libre de Bruxelles
- **Academic Year:** 2025-2026

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{pintoickowicz2026privacy,
  title={Privacy-Preserving Age Verification under GDPR: 
         AI-Based Identity Extraction and Encrypted Token Exchange},
  author={Pinto Ickowicz, Priscila},
  year={2026},
  school={Universit{\'e} Libre de Bruxelles}
}
```
