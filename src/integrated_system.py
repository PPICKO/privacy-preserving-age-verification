"""
Integrated System Module
Connects ID detection pipeline with cryptographic token generation.

For thesis: "Privacy-Preserving Age Verification under GDPR"
Author: Priscila PINTO ICKOWICZ

This module integrates:
1. ID detection (YOLOv8 + OCR)
2. Token generation (JWT + RSA)
3. GDPR-compliant data handling

Usage:
    python integrated_system.py --demo
    python integrated_system.py --audience "website.com"
"""

import sys
import shutil
import logging
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

# Import token system
try:
    from token_system import AgeVerificationToken, TokenConfig, TokenValidator
    TOKEN_IMPORT_SUCCESS = True
except ImportError:
    TOKEN_IMPORT_SUCCESS = False
    print("Warning: token_system.py not found in current directory")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IntegratedSystemConfig:
    """Combined configuration for detection + token system"""
    # Detection settings
    model_path: str = r"C:\Users\branq\Desktop\thesis\my_model.pt"
    camera_index: int = 0
    max_retry_attempts: int = 5
    detection_threshold: float = 0.80
    
    # Token settings
    token_validity_days: int = 30
    enable_device_binding: bool = True
    include_names_in_token: bool = False  # Privacy: names optional
    token_issuer: str = "AgeVerification-Thesis-System"
    
    # Output settings
    output_dir: str = "outputs"
    
    # Privacy settings (GDPR compliance)
    delete_snapshots_after_token: bool = True  # Immediate deletion
    include_age_value: bool = False  # Only store is_adult boolean
    
    # Logging
    log_level: str = "INFO"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AgeInfo:
    """Age calculation result (matches id_detection.py)"""
    dob_parsed: Optional[str] = None
    age_years: Optional[int] = None
    status: str = "unknown"  # "adult", "minor", or "unknown"
    
    def is_adult(self) -> bool:
        return self.age_years is not None and self.age_years >= 18


@dataclass
class VerificationResult:
    """Complete verification result"""
    success: bool = False
    is_adult: Optional[bool] = None
    age_years: Optional[int] = None
    token_data: Optional[Dict] = None
    message: str = ""
    error: Optional[str] = None


# ============================================================================
# INTEGRATED SYSTEM
# ============================================================================

class PrivacyPreservingAgeVerification:
    """
    Complete privacy-preserving age verification system.
    
    Workflow:
    1. Capture ID from webcam (detection module)
    2. Detect DOB + names using YOLOv8 + OCR
    3. Calculate age
    4. If verified: generate encrypted token
    5. IMMEDIATELY delete all ID images and OCR data
    6. Return token to user (QR code + text)
    """
    
    def __init__(self, config: IntegratedSystemConfig = None):
        self.config = config or IntegratedSystemConfig()
        self.logger = self._setup_logging()
        self.token_generator = None
        
        # Initialize token generator
        self._setup_token_generator()
        
        self.logger.info("Privacy-Preserving Age Verification System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_token_generator(self):
        """Initialize token generation system"""
        if not TOKEN_IMPORT_SUCCESS:
            self.logger.error("Token system not available")
            return
        
        token_config = TokenConfig(
            VALIDITY_DAYS=self.config.token_validity_days,
            ENABLE_DEVICE_BINDING=self.config.enable_device_binding,
            INCLUDE_NAME_IN_TOKEN=self.config.include_names_in_token,
            INCLUDE_AGE_VALUE=self.config.include_age_value,
        )
        self.token_generator = AgeVerificationToken(token_config)
    
    def process_verification(
        self,
        age_info: AgeInfo,
        ocr_results: Dict,
        snap_dir: Optional[Path] = None,
        audience: str = "default"
    ) -> VerificationResult:
        """
        Process verification result and generate token.
        
        Args:
            age_info: AgeInfo object with age calculation results
            ocr_results: OCR results (names, etc.)
            snap_dir: Snapshot directory to delete (optional)
            audience: Target website/service identifier
        
        Returns:
            VerificationResult with token or error
        """
        result = VerificationResult()
        
        try:
            # Check if age was successfully determined
            if age_info.status == 'unknown':
                self.logger.warning("Age determination failed - cannot issue token")
                result.message = "Could not determine age from ID"
                return result
            
            # Check token generator
            if self.token_generator is None:
                self.logger.error("Token generator not initialized")
                result.error = "Token system not available"
                return result
            
            # Determine if user is adult
            is_adult = (age_info.status == 'adult')
            
            # Extract names (if configured)
            given_name = None
            surname = None
            if self.config.include_names_in_token and ocr_results:
                given_name = ocr_results.get('GivenName', [None])[0]
                surname = ocr_results.get('Surname', [None])[0]
            
            # Generate token
            self.logger.info(f"Generating token for {'ADULT' if is_adult else 'MINOR'}")
            token_data = self.token_generator.generate_token(
                is_adult=is_adult,
                age_years=age_info.age_years,
                given_name=given_name,
                surname=surname,
                audience=audience
            )
            
            # CRITICAL: Delete sensitive data immediately (GDPR compliance)
            if self.config.delete_snapshots_after_token and snap_dir:
                self._secure_delete(snap_dir)
            
            # Build result
            result.success = True
            result.is_adult = is_adult
            result.age_years = age_info.age_years
            result.token_data = token_data
            result.message = "Age verified successfully. Token generated."
            
            self.logger.info(f"✓ Token issued: {token_data['token_id'][:16]}...")
            
        except Exception as e:
            self.logger.error(f"Verification error: {e}", exc_info=True)
            result.error = str(e)
        
        return result
    
    def _secure_delete(self, path: Path):
        """
        Securely delete directory containing sensitive data.
        
        Implements GDPR storage limitation:
        - No images persisted
        - No OCR text stored
        - Only token remains
        """
        if path and path.exists():
            try:
                shutil.rmtree(path)
                self.logger.info(f"Deleted sensitive data: {path}")
            except Exception as e:
                self.logger.error(f"Delete failed: {e}")
    
    def display_token(self, token_data: Dict):
        """Display token information to user"""
        print("\n" + "=" * 70)
        print("                   AGE VERIFICATION SUCCESSFUL")
        print("=" * 70)
        print(f"\n Your age has been verified")
        print(f"Token valid until: {token_data['expires_at']}")
        print(f"Token ID: {token_data['token_id'][:16]}...")
        
        if 'qr_code_path' in token_data:
            print(f"\n  QR Code saved: {token_data['qr_code_path']}")
        
        print(f"\n  You can use this token for {token_data['validity_days']} days.")
        
        print("\n PRIVACY NOTICE:")
        print(" All ID images have been DELETED")
        print(" No personal data is stored")
        print(" Token contains only verification status")
        print(" Token is device-specific")
        print("=" * 70 + "\n")
    
    def get_public_key(self) -> Optional[str]:
        """Get public key for third-party validation"""
        if self.token_generator:
            return self.token_generator.get_public_key_for_verification()
        return None


# ============================================================================
# DEMO WORKFLOW
# ============================================================================

def run_demo():
    """
    Demonstrate complete workflow:
    1. Simulate ID detection
    2. Generate token
    3. Validate token
    """
    print("\n" + "=" * 70)
    print("   DEMO: Privacy-Preserving Age Verification")
    print("=" * 70)
    
    if not TOKEN_IMPORT_SUCCESS:
        print("\n Cannot run demo: token_system.py not found")
        print("Make sure token_system.py is in the same directory")
        return
    
    # Step 1: Simulate detection result
    print("\n[1] ID Detection (Simulated)")
    print("Simulating YOLOv8 + OCR detection...")
    
    age_info = AgeInfo(
        dob_parsed="1990-05-15",
        age_years=34,
        status="adult"
    )
    
    ocr_results = {
        "GivenName": ["John"],
        "Surname": ["Doe"],
        "DOB": ["15/05/1990"]
    }
    
    print(f" DOB detected: {age_info.dob_parsed}")
    print(f" Age calculated: {age_info.age_years} years")
    print(f" Status: {age_info.status.upper()}")
    
    # Step 2: Generate token
    print("\n[2] Token Generation")
    
    config = IntegratedSystemConfig(
        token_validity_days=30,
        enable_device_binding=True,
        include_names_in_token=False,  # Maximum privacy
        delete_snapshots_after_token=True
    )
    
    system = PrivacyPreservingAgeVerification(config)
    
    result = system.process_verification(
        age_info=age_info,
        ocr_results=ocr_results,
        snap_dir=None,  # No actual files in demo
        audience="adult-website.com"
    )
    
    if result.success:
        system.display_token(result.token_data)
        token = result.token_data["token"]
    else:
        print(f" Verification failed: {result.error or result.message}")
        return
    
    # Step 3: Validate token (simulating website)
    print("\n[3] Website Token Validation")
    print("    User presents token to adult-website.com...")
    
    public_key = system.get_public_key()
    validator = TokenValidator(
        public_key_pem=public_key,
        expected_issuer=config.token_issuer
    )
    
    validation = validator.validate_token(
        token=token,
        audience="adult-website.com"
    )
    
    if validation["valid"] and validation["access_granted"]:
        print(" Token validated successfully")
        print(" Access GRANTED to age-restricted content")
        print(" Website learned only: 'user is adult'")
    else:
        print(f"Access denied: {validation.get('reason')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("   WORKFLOW COMPLETE - GDPR COMPLIANCE")
    print("=" * 70)
    print("\n No ID images stored")
    print(" No DOB stored")
    print(" Token contains only 'is_adult' boolean")
    print(" Token is device-bound")
    print(" Token expires in 30 days")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Integrated Age Verification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrated_system.py --demo
  python integrated_system.py --audience "my-website.com"
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration workflow"
    )
    parser.add_argument(
        "--audience",
        type=str,
        default="default",
        help="Target audience for token (default: 'default')"
    )
    parser.add_argument(
        "--validity-days",
        type=int,
        default=30,
        help="Token validity in days (default: 30)"
    )
    parser.add_argument(
        "--include-names",
        action="store_true",
        help="Include names in token (reduces privacy)"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        print("\nUsage: python integrated_system.py --demo")
        print("\nThis module is designed to be imported by id_detection.py")
        print("Run --demo to see the complete workflow demonstration")


if __name__ == "__main__":
    main()
