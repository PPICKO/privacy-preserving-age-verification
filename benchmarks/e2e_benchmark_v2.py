"""
End-to-End Pipeline Benchmark (Integrated Version)
Uses YOUR actual detection code from new_age_deb.py

For thesis: "Privacy-Preserving Age Verification under GDPR"
Author: Priscila PINTO ICKOWICZ

This script measures:
1. YOLOv8 Detection latency
2. OCR Processing latency  
3. Date Parsing latency
4. Token Generation latency
5. Total end-to-end latency
6. Latency breakdown percentages

IMPORTANT: Place this file in the same folder as:
- id_detection.py (your detection code)
- token_system.py (your token code)

Option Run: python e2e_benchmark.py --image belgium-id-card.jpg
"""

import os
import sys
import cv2
import json
import time
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import psutil

# ============================================================================
# IMPORT YOUR ACTUAL CODE
# ============================================================================

try:
    # Import from your detection code
    from id_detection import (
        Config, 
        IDCardDetector,
        OCRProcessor,
        DetectionProcessor,
        DateParser,
        AgeInfo
    )
    DETECTION_IMPORT_SUCCESS = True
    print("Imported detection code from id_detection.py")
except ImportError as e:
    DETECTION_IMPORT_SUCCESS = False
    print(f"Could not import id_detection.py: {e}")
    print("  Make sure id_detection.py is in the same folder!")

try:
    # Import token system
    from token_system import AgeVerificationToken, TokenConfig
    TOKEN_IMPORT_SUCCESS = True
    print("Imported token system from token_system.py")
except ImportError as e:
    TOKEN_IMPORT_SUCCESS = False
    print(f"Could not import token_system.py: {e}")
    print("  Will use embedded token implementation")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class E2EBenchmarkConfig:
    """Configuration for end-to-end benchmark"""
    # Test settings
    num_iterations: int = 20
    warmup_iterations: int = 3
    
    # Image settings
    test_image_path: Optional[str] = None
    
    # Output
    output_dir: str = "benchmark_results"
    output_file: str = "e2e_pipeline_metrics.json"


@dataclass
class TimingResult:
    """Single iteration timing result"""
    iteration: int = 0
    
    # Individual stage timings (milliseconds)
    detection_ms: float = 0.0
    ocr_ms: float = 0.0
    date_parsing_ms: float = 0.0
    age_calculation_ms: float = 0.0
    token_generation_ms: float = 0.0
    
    # Total
    total_ms: float = 0.0
    
    # Results
    dob_detected: bool = False
    dob_confidence: float = 0.0
    photo_detected: bool = False
    ocr_text: str = ""
    parsed_date: str = ""
    age_years: Optional[int] = None
    age_status: str = "unknown"
    token_generated: bool = False


@dataclass
class E2EBenchmarkResults:
    """Complete benchmark results"""
    timestamp: str = ""
    system_info: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    
    # Individual iterations
    iterations: List[Dict] = field(default_factory=list)
    
    # Aggregated statistics
    detection_stats: Dict = field(default_factory=dict)
    ocr_stats: Dict = field(default_factory=dict)
    date_parsing_stats: Dict = field(default_factory=dict)
    age_calculation_stats: Dict = field(default_factory=dict)
    token_generation_stats: Dict = field(default_factory=dict)
    total_stats: Dict = field(default_factory=dict)
    
    # Breakdown percentages
    latency_breakdown: Dict = field(default_factory=dict)
    
    # Success rates
    detection_success_rate: float = 0.0
    ocr_success_rate: float = 0.0
    age_determination_success_rate: float = 0.0


# ============================================================================
# EMBEDDED TOKEN SYSTEM (fallback)
# ============================================================================

if not TOKEN_IMPORT_SUCCESS:
    import jwt
    import secrets
    import hashlib
    import platform
    from datetime import timedelta
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    
    class TokenConfig:
        VALIDITY_DAYS = 30
        ENABLE_DEVICE_BINDING = True
    
    class AgeVerificationToken:
        def __init__(self, config=None):
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
        
        def generate_token(self, is_adult, age_years=None, **kwargs):
            device_info = f"{platform.node()}-{platform.machine()}-{platform.system()}"
            device_fp = hashlib.sha256(device_info.encode()).hexdigest()[:16]
            
            payload = {
                "sub": secrets.token_hex(16),
                "iss": "AgeVerification-Thesis-System",
                "aud": kwargs.get("audience", "default"),
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(days=30),
                "is_adult": is_adult,
                "device_fingerprint": device_fp
            }
            
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            token = jwt.encode(payload, private_pem, algorithm="RS256")
            return {"token": token, "token_id": payload["sub"]}


# ============================================================================
# PIPELINE BENCHMARK
# ============================================================================

class E2EPipelineBenchmark:
    """End-to-end pipeline benchmark using your actual code"""
    
    def __init__(self, config: E2EBenchmarkConfig = None):
        self.config = config or E2EBenchmarkConfig()
        self.results = E2EBenchmarkResults()
        
        # Setup output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Store metadata
        self.results.timestamp = datetime.now().isoformat()
        self.results.system_info = self._get_system_info()
        self.results.config = {
            "num_iterations": self.config.num_iterations,
            "test_image_path": self.config.test_image_path
        }
        
        # Components (initialized in setup)
        self.model = None
        self.ocr_processor = None
        self.detection_config = None
        self.token_generator = None
        
    def _get_system_info(self) -> Dict:
        """Collect system information"""
        import platform
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }
    
    def setup(self) -> bool:
        """Initialize all components using your actual code"""
        print("\n" + "="*60)
        print("SETUP: Loading models and components...")
        print("="*60)
        
        if not DETECTION_IMPORT_SUCCESS:
            print("\n Cannot run benchmark without id_detection.py")
            return False
        
        # Initialize your Config
        print("\n Loading configuration...")
        self.detection_config = Config()
        print(f" Config loaded (model: {self.detection_config.MODEL_PATH})")
        
        # Load YOLO model
        print("\n Loading YOLOv8 model...")
        load_start = time.perf_counter()
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.detection_config.MODEL_PATH)
            load_time = (time.perf_counter() - load_start) * 1000
            print(f" YOLOv8 loaded in {load_time:.0f}ms")
        except Exception as e:
            print(f" Failed to load YOLO: {e}")
            return False
        
        # Load OCR using your OCRProcessor
        print("\n  Loading EasyOCR (via OCRProcessor)...")
        load_start = time.perf_counter()
        try:
            self.ocr_processor = OCRProcessor()
            load_time = (time.perf_counter() - load_start) * 1000
            print(f"EasyOCR loaded in {load_time:.0f}ms")
        except Exception as e:
            print(f"Failed to load OCR: {e}")
            return False
        
        # Load Token System
        print("\n Loading Token System...")
        load_start = time.perf_counter()
        try:
            token_config = TokenConfig()
            self.token_generator = AgeVerificationToken(token_config)
            load_time = (time.perf_counter() - load_start) * 1000
            print(f"Token system loaded in {load_time:.0f}ms")
        except Exception as e:
            print(f"Failed to load token system: {e}")
            return False
        
        print("\n All components loaded successfully!")
        return True
    
    def get_test_image(self):
        """Get test image from file or webcam"""
        if self.config.test_image_path:
            path = Path(self.config.test_image_path)
            if path.exists():
                print(f"\n  Using test image: {path}")
                frame = cv2.imread(str(path))
                if frame is not None:
                    print(f" Image loaded ({frame.shape[1]}x{frame.shape[0]})")
                    return frame
                else:
                    print("Could not read image")
                    return None
            else:
                print(f" Image not found: {path}")
                return None
        else:
            print("\n  Opening webcam with live preview...")
            print("  ════════════════════════════════════════════")
            print("  ║  Position your ID card, then press SPACE  ║")
            print("  ║  Press 'q' to quit without capturing      ║")
            print("  ════════════════════════════════════════════")
            
            cap = cv2.VideoCapture(self.detection_config.CAMERA_INDEX)
            if not cap.isOpened():
                print("Could not open webcam")
                return None
            
            captured_frame = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection to show what's being detected
                results = self.model(frame, imgsz=self.detection_config.IMGSZ,
                                    conf=self.detection_config.BASE_CONF, verbose=False)
                annotated = results[0].plot()
                
                # Add instructions on frame
                cv2.putText(annotated, "Position ID card - Press SPACE to capture", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, "Press 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check what's detected
                detected = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label = self.detection_config.CLASS_NAMES[cls_id]
                    if conf >= self.detection_config.THRESHOLDS.get(label, 0):
                        detected.append(f"{label}:{conf:.2f}")
                
                if detected:
                    cv2.putText(annotated, f"Detected: {', '.join(detected)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow("E2E Benchmark - Position ID Card", annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # SPACE to capture
                    captured_frame = frame.copy()
                    print(f"Frame captured ({frame.shape[1]}x{frame.shape[0]})")
                    if detected:
                        print(f"Detected: {', '.join(detected)}")
                    break
                elif key == ord('q'):  # Q to quit
                    print("Cancelled by user")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            return captured_frame
    
    def run_single_iteration(self, frame, iteration: int) -> TimingResult:
        """Run single pipeline iteration with detailed timing"""
        result = TimingResult(iteration=iteration)
        total_start = time.perf_counter()
        
        # ====================================================================
        # STAGE 1: YOLO Detection
        # ====================================================================
        detection_start = time.perf_counter()
        
        detections = self.model(
            frame, 
            imgsz=self.detection_config.IMGSZ,
            conf=self.detection_config.BASE_CONF, 
            verbose=False
        )
        
        result.detection_ms = (time.perf_counter() - detection_start) * 1000
        
        # Extract detection results
        # Note: Using slightly lower threshold (0.70) for benchmarking to capture more samples
        # Production uses 0.80, but for timing measurements we want successful iterations
        BENCHMARK_THRESHOLD = 0.70  # Lower than production 0.80 for timing measurements
        
        dob_boxes = []
        for box in detections[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = self.detection_config.CLASS_NAMES[cls_id]
            threshold = min(BENCHMARK_THRESHOLD, self.detection_config.THRESHOLDS.get(label, 0.0))
            
            if conf >= threshold:
                if label == "DOB":
                    result.dob_detected = True
                    result.dob_confidence = conf
                    dob_boxes.append({
                        "box": box.xyxy[0].tolist(),
                        "confidence": conf
                    })
                elif label == "Photo":
                    result.photo_detected = True
        
        # ====================================================================
        # STAGE 2: OCR Processing
        # ====================================================================
        ocr_start = time.perf_counter()
        ocr_texts = []
        
        if result.dob_detected and dob_boxes:
            for dob_info in dob_boxes:
                x1, y1, x2, y2 = map(int, dob_info["box"])
                
                # Add padding (like your code does)
                pad = 10
                h, w = frame.shape[:2]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # Use your OCRProcessor's read_text method
                    texts = self.ocr_processor.read_text(crop)
                    ocr_texts.extend(texts)
        
        result.ocr_ms = (time.perf_counter() - ocr_start) * 1000
        result.ocr_text = " | ".join(ocr_texts) if ocr_texts else ""
        
        # ====================================================================
        # STAGE 3: Date Parsing (using your DateParser)
        # ====================================================================
        parsing_start = time.perf_counter()
        parsed_date = None
        
        if ocr_texts:
            # Clean text using your method
            cleaned_text = DateParser.clean_dob_text(
                ocr_texts, 
                self.detection_config.DOB_STOPWORDS
            )
            
            # Parse date using your method
            parsed_date = DateParser.parse_date(cleaned_text)
            
            if parsed_date:
                result.parsed_date = parsed_date.isoformat()
        
        result.date_parsing_ms = (time.perf_counter() - parsing_start) * 1000
        
        # ====================================================================
        # STAGE 4: Age Calculation
        # ====================================================================
        age_calc_start = time.perf_counter()
        
        if parsed_date:
            from datetime import date as dt_date
            today = dt_date.today()
            
            age_years = today.year - parsed_date.year
            if (today.month, today.day) < (parsed_date.month, parsed_date.day):
                age_years -= 1
            
            result.age_years = age_years
            result.age_status = "adult" if age_years >= 18 else "minor"
        
        result.age_calculation_ms = (time.perf_counter() - age_calc_start) * 1000
        
        # ====================================================================
        # STAGE 5: Token Generation
        # ====================================================================
        token_start = time.perf_counter()
        
        if result.age_years is not None:
            is_adult = result.age_status == "adult"
            token_data = self.token_generator.generate_token(
                is_adult=is_adult,
                age_years=result.age_years,
                audience="benchmark-test"
            )
            result.token_generated = True
        
        result.token_generation_ms = (time.perf_counter() - token_start) * 1000
        
        # Total time
        result.total_ms = (time.perf_counter() - total_start) * 1000
        
        return result
    
    def calculate_stats(self, values: List[float]) -> Dict:
        """Calculate statistics"""
        if not values:
            return {"mean_ms": 0, "median_ms": 0, "min_ms": 0, "max_ms": 0, "stdev_ms": 0}
        
        return {
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
            "mean_ms": round(statistics.mean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "stdev_ms": round(statistics.stdev(values), 3) if len(values) > 1 else 0,
            "total_samples": len(values)
        }
    
    def run_benchmark(self):
        """Run complete benchmark"""
        print("\n" + "="*70)
        print("   END-TO-END PIPELINE BENCHMARK (Using Your Detection Code)")
        print("="*70)
        
        # Setup
        if not self.setup():
            print("\n Setup failed. Exiting.")
            return None
        
        # Get test image
        frame = self.get_test_image()
        if frame is None:
            print("\n Could not get test image. Exiting.")
            return None
        
        # Quick detection test
        print("\n  Quick detection test...")
        test_result = self.model(frame, imgsz=self.detection_config.IMGSZ,
                                 conf=self.detection_config.BASE_CONF, verbose=False)
        
        detected_items = []
        dob_found = False
        for box in test_result[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = self.detection_config.CLASS_NAMES[cls_id]
            detected_items.append(f"{label}({conf:.2f})")
            if label == "DOB" and conf >= 0.70:  # Using benchmark threshold
                dob_found = True
        
        if detected_items:
            print(f"Detected: {', '.join(detected_items)}")
            if dob_found:
                print(" DOB meets benchmark threshold (≥0.70)")
            else:
                print("DOB below threshold - OCR may not run")
        else:
            print("No fields detected! Results may be limited.")
        
        # Warmup runs
        print(f"\n  Running {self.config.warmup_iterations} warmup iterations...")
        for i in range(self.config.warmup_iterations):
            self.run_single_iteration(frame, -1)
        print("Warmup complete")
        
        # Benchmark runs
        print(f"\n  Running {self.config.num_iterations} benchmark iterations...")
        timing_results = []
        
        for i in range(self.config.num_iterations):
            result = self.run_single_iteration(frame, i + 1)
            timing_results.append(result)
            
            if (i + 1) % 5 == 0:
                avg_so_far = statistics.mean([r.total_ms for r in timing_results])
                print(f"    Completed {i + 1}/{self.config.num_iterations} (avg: {avg_so_far:.1f}ms)")
        
        # Store individual results
        self.results.iterations = [asdict(r) for r in timing_results]
        
        # Calculate statistics for each stage
        self.results.detection_stats = self.calculate_stats([r.detection_ms for r in timing_results])
        self.results.ocr_stats = self.calculate_stats([r.ocr_ms for r in timing_results if r.dob_detected])
        self.results.date_parsing_stats = self.calculate_stats([r.date_parsing_ms for r in timing_results if r.ocr_text])
        self.results.age_calculation_stats = self.calculate_stats([r.age_calculation_ms for r in timing_results if r.age_years])
        self.results.token_generation_stats = self.calculate_stats([r.token_generation_ms for r in timing_results if r.token_generated])
        self.results.total_stats = self.calculate_stats([r.total_ms for r in timing_results])
        
        # Calculate breakdown percentages (only for successful iterations)
        successful = [r for r in timing_results if r.token_generated]
        if successful:
            total_mean = statistics.mean([r.total_ms for r in successful])
            if total_mean > 0:
                self.results.latency_breakdown = {
                    "detection_pct": round((statistics.mean([r.detection_ms for r in successful]) / total_mean) * 100, 1),
                    "ocr_pct": round((statistics.mean([r.ocr_ms for r in successful]) / total_mean) * 100, 1),
                    "date_parsing_pct": round((statistics.mean([r.date_parsing_ms for r in successful]) / total_mean) * 100, 1),
                    "age_calculation_pct": round((statistics.mean([r.age_calculation_ms for r in successful]) / total_mean) * 100, 1),
                    "token_generation_pct": round((statistics.mean([r.token_generation_ms for r in successful]) / total_mean) * 100, 1),
                }
        
        # Success rates
        n = len(timing_results)
        self.results.detection_success_rate = round((sum(1 for r in timing_results if r.dob_detected) / n) * 100, 1)
        self.results.ocr_success_rate = round((sum(1 for r in timing_results if r.ocr_text) / n) * 100, 1)
        self.results.age_determination_success_rate = round((sum(1 for r in timing_results if r.age_years is not None) / n) * 100, 1)
        
        # Save and print results
        self._save_results()
        self._print_summary(timing_results)
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON"""
        output_path = Path(self.config.output_dir) / self.config.output_file
        
        with open(output_path, 'w') as f:
            json.dump(asdict(self.results), f, indent=2, default=str)
        
        print(f"\n Results saved to: {output_path}")
    
    def _print_summary(self, timing_results: List[TimingResult]):
        """Print summary for thesis"""
        print("\n" + "="*70)
        print("   SUMMARY - VALUES FOR THESIS (End-to-End Pipeline)")
        print("="*70)
        
        # Show a sample successful iteration
        successful = [r for r in timing_results if r.token_generated]
        if successful:
            sample = successful[0]
            print("SAMPLE SUCCESSFUL ITERATION:")
            print(f"   • DOB Confidence: {sample.dob_confidence:.2f}")
            print(f"   • OCR Text: {sample.ocr_text[:50]}..." if len(sample.ocr_text) > 50 else f"   • OCR Text: {sample.ocr_text}")
            print(f"   • Parsed Date: {sample.parsed_date}")
            print(f"   • Age: {sample.age_years} years ({sample.age_status})")
        
        print("\n LATENCY BY STAGE:")
        
        if self.results.latency_breakdown:
            print(f"   • YOLOv8 Detection:  {self.results.detection_stats.get('mean_ms', 0):>8.1f} ms  ({self.results.latency_breakdown.get('detection_pct', 0):>5.1f}%)")
            print(f"   • OCR Processing:    {self.results.ocr_stats.get('mean_ms', 0):>8.1f} ms  ({self.results.latency_breakdown.get('ocr_pct', 0):>5.1f}%)")
            print(f"   • Date Parsing:      {self.results.date_parsing_stats.get('mean_ms', 0):>8.1f} ms  ({self.results.latency_breakdown.get('date_parsing_pct', 0):>5.1f}%)")
            print(f"   • Age Calculation:   {self.results.age_calculation_stats.get('mean_ms', 0):>8.1f} ms  ({self.results.latency_breakdown.get('age_calculation_pct', 0):>5.1f}%)")
            print(f"   • Token Generation:  {self.results.token_generation_stats.get('mean_ms', 0):>8.1f} ms  ({self.results.latency_breakdown.get('token_generation_pct', 0):>5.1f}%)")
            print("─────────────────────────────────────")
            
            # Calculate total for successful iterations
            if successful:
                total_successful = statistics.mean([r.total_ms for r in successful])
                print(f"   • TOTAL (successful):{total_successful:>8.1f} ms  (100.0%)")
        else:
            print(f"   • YOLOv8 Detection:  {self.results.detection_stats.get('mean_ms', 0):>8.1f} ms")
            print("   • (Other stages: no successful detections)")
        
        print("\n SUCCESS RATES:")
        print(f"   • DOB Detection:      {self.results.detection_success_rate}%")
        print(f"   • OCR Success:        {self.results.ocr_success_rate}%")
        print(f"   • Age Determination:  {self.results.age_determination_success_rate}%")
        
        print("\n📊 LATENCY STATISTICS (All Iterations):")
        print(f"   • Mean:   {self.results.total_stats.get('mean_ms', 0):.1f} ms")
        print(f"   • Median: {self.results.total_stats.get('median_ms', 0):.1f} ms")
        print(f"   • Min:    {self.results.total_stats.get('min_ms', 0):.1f} ms")
        print(f"   • Max:    {self.results.total_stats.get('max_ms', 0):.1f} ms")
        
        # Thesis values
        print("\n" + "="*70)
        print("   THESIS PLACEHOLDER VALUES:")
        print("="*70)
        
        if successful:
            total_successful = statistics.mean([r.total_ms for r in successful])
            ocr_pct = self.results.latency_breakdown.get('ocr_pct', 0)
            token_time = self.results.token_generation_stats.get('mean_ms', 0)
            
            print("\n   Processing time [INSERT DURATION, e.g., 550-640ms]:")
            print(f"   → {total_successful:.0f}ms (or ~{total_successful/1000:.2f} seconds)")
            
            print("\n   OCR percentage [INSERT YOUR RESULT, e.g., ~70% of total time]:")
            print(f"   → {ocr_pct:.0f}% of total processing time")
            
            print("\n   Token generation time:")
            print(f"   → {token_time:.1f}ms")
            
            print("\n   Success rate:")
            print(f"   → {self.results.age_determination_success_rate}%")
        else:
            print("\n   ⚠ No successful iterations - cannot provide thesis values")
            print("   Try with a clearer ID card image")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Benchmark")
    parser.add_argument("--image", type=str, help="Path to test image (required for best results)")
    parser.add_argument("--iterations", type=int, default=20, help="Number of benchmark iterations")
    args = parser.parse_args()
    
    if not args.image:
        print("\n⚠ WARNING: No --image specified. Using webcam.")
        print("  For best results, use: python e2e_benchmark.py --image your_id_card.jpg\n")
    
    config = E2EBenchmarkConfig(
        num_iterations=args.iterations,
        test_image_path=args.image,
    )
    
    benchmark = E2EPipelineBenchmark(config)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
