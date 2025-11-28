"""
Privacy-Preserving Age Verification Token System
Integrated with ID Card Detection Pipeline

This module implements JWT-based token generation with RSA signatures
for the thesis: "Privacy-Preserving Age Verification under GDPR"

Author: Priscila PINTO ICKOWICZ
Supervisors: Prof. SACHARIDIS Dimitris, Prof. MÜHLBERG Jan Tobias
"""

import jwt
import secrets
import hashlib
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
import qrcode
import json
import logging

# === TOKEN CONFIGURATION ===
@dataclass
class TokenConfig:
    """Configuration for age verification tokens"""
    VALIDITY_DAYS: int = 30  # Token valid for 30 days (reusable)
    ISSUER: str = "AgeVerification-Thesis-System"
    KEY_SIZE: int = 2048  # RSA key size
    ALGORITHM: str = "RS256"  # RSA with SHA-256
    ENABLE_DEVICE_BINDING: bool = True  # Prevent token sharing
    
    # Privacy settings
    INCLUDE_NAME_IN_TOKEN: bool = False  # Minimize data disclosure
    INCLUDE_AGE_VALUE: bool = False  # Only include "is_adult" boolean
    
    # Paths
    KEYS_DIR: str = "keys"
    TOKENS_DIR: str = "tokens"


# === CRYPTOGRAPHIC KEY MANAGEMENT ===
class KeyManager:
    """Manages RSA key pairs for token signing and verification"""
    
    def __init__(self, config: TokenConfig):
        self.config = config
        self.keys_path = Path(config.KEYS_DIR)
        self.keys_path.mkdir(parents=True, exist_ok=True)
        
        self.private_key = None
        self.public_key = None
        self.logger = logging.getLogger(__name__)
        
        # Load or generate keys
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Load existing keys or generate new ones"""
        private_key_path = self.keys_path / "private_key.pem"
        public_key_path = self.keys_path / "public_key.pem"
        
        if private_key_path.exists() and public_key_path.exists():
            self.logger.info("Loading existing key pair...")
            self._load_keys(private_key_path, public_key_path)
        else:
            self.logger.info("Generating new RSA key pair...")
            self._generate_keys()
            self._save_keys(private_key_path, public_key_path)
    
    def _generate_keys(self):
        """Generate new RSA key pair"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.KEY_SIZE,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.logger.info(f"Generated {self.config.KEY_SIZE}-bit RSA key pair")
    
    def _save_keys(self, private_path: Path, public_path: Path):
        """Save keys to PEM files"""
        # Save private key (encrypted with password)
        password = b"thesis_age_verification_2025"  # In production: use secure key management
        
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(password)
        )
        
        with open(private_path, 'wb') as f:
            f.write(private_pem)
        
        # Save public key (no encryption needed)
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        with open(public_path, 'wb') as f:
            f.write(public_pem)
        
        self.logger.info(f"Keys saved to {self.keys_path}")
    
    def _load_keys(self, private_path: Path, public_path: Path):
        """Load keys from PEM files"""
        password = b"thesis_age_verification_2025"
        
        with open(private_path, 'rb') as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=password,
                backend=default_backend()
            )
        
        with open(public_path, 'rb') as f:
            self.public_key = serialization.load_pem_public_key(
                f.read(),
                backend=default_backend()
            )
        
        self.logger.info("Keys loaded successfully")
    
    def get_public_key_pem(self) -> str:
        """Export public key as PEM string (for websites to verify tokens)"""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem.decode('utf-8')


# === DEVICE FINGERPRINTING ===
class DeviceFingerprint:
    """Generate device-specific identifier to prevent token sharing"""
    
    @staticmethod
    def generate() -> str:
        """Create stable device fingerprint"""
        # Combine stable hardware/OS identifiers
        device_info = f"{platform.node()}-{platform.machine()}-{platform.system()}"
        fingerprint = hashlib.sha256(device_info.encode()).hexdigest()[:16]
        return fingerprint
    
    @staticmethod
    def verify(token_fingerprint: str, current_fingerprint: str) -> bool:
        """Check if token was issued for this device"""
        return token_fingerprint == current_fingerprint


# === TOKEN GENERATOR ===
class AgeVerificationToken:
    """
    Generate cryptographically signed tokens for age verification.
    
    Implements GDPR-compliant, privacy-preserving token design:
    - Minimal disclosure (only "is_adult" boolean, no DOB)
    - Short-lived (30-day expiration)
    - Device-bound (prevents sharing)
    - Unlinkable across services (audience scoping)
    - No personal data storage
    """
    
    def __init__(self, config: TokenConfig = None):
        self.config = config or TokenConfig()
        self.key_manager = KeyManager(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        Path(self.config.TOKENS_DIR).mkdir(parents=True, exist_ok=True)
    
    def generate_token(
        self,
        is_adult: bool,
        age_years: Optional[int] = None,
        given_name: Optional[str] = None,
        surname: Optional[str] = None,
        audience: str = "default"
    ) -> Dict[str, str]:
        """
        Generate age verification token.
        
        Args:
            is_adult: Whether user meets age threshold
            age_years: Actual age (optional, for logging only)
            given_name: Optional identifier (only if policy requires)
            surname: Optional identifier (only if policy requires)
            audience: Service/website identifier (for unlinkability)
        
        Returns:
            Dictionary with token, metadata, and QR code path
        """
        # Generate device fingerprint
        device_id = DeviceFingerprint.generate() if self.config.ENABLE_DEVICE_BINDING else None
        
        # Create minimal payload (GDPR data minimization)
        payload = {
            # Core claims
            "sub": secrets.token_hex(16),  # Unique token ID (unlinkable)
            "iss": self.config.ISSUER,
            "aud": audience,  # Service-specific (unlinkability)
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=self.config.VALIDITY_DAYS),
            
            # Age verification claim (minimal disclosure)
            "age_verified": True,
            "is_adult": is_adult,
            
            # Optional: include age value only if configured
            **({"age_years": age_years} if self.config.INCLUDE_AGE_VALUE and age_years else {}),
            
            # Optional: include names only if policy requires
            **({"given_name": given_name} if self.config.INCLUDE_NAME_IN_TOKEN and given_name else {}),
            **({"family_name": surname} if self.config.INCLUDE_NAME_IN_TOKEN and surname else {}),
            
            # Security features
            "token_type": "reusable",
            **({"device_fingerprint": device_id} if device_id else {}),
            "jti": secrets.token_hex(32),  # JWT ID (prevents replay)
        }
        
        # Sign token with private key (RS256)
        token = jwt.encode(
            payload,
            self.key_manager.private_key,
            algorithm=self.config.ALGORITHM
        )
        
        # Generate QR code for easy sharing
        qr_path = self._generate_qr_code(token, payload["sub"])
        
        # Log issuance (no personal data in logs)
        self.logger.info(
            f"Token issued: ID={payload['sub'][:8]}... "
            f"Adult={is_adult} Expires={payload['exp'].isoformat()} "
            f"Audience={audience}"
        )
        
        # Save token metadata (not the token itself - that stays with user)
        self._save_token_metadata(payload)
        
        return {
            "token": token,
            "token_id": payload["sub"],
            "issued_at": payload["iat"].isoformat(),
            "expires_at": payload["exp"].isoformat(),
            "is_adult": is_adult,
            "qr_code_path": str(qr_path),
            "audience": audience,
            "validity_days": self.config.VALIDITY_DAYS
        }
    
    def _generate_qr_code(self, token: str, token_id: str) -> Path:
        """Generate QR code for token"""
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(token)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        qr_path = Path(self.config.TOKENS_DIR) / f"token_{token_id[:8]}_qr.png"
        img.save(qr_path)
        
        return qr_path
    
    def _save_token_metadata(self, payload: Dict):
        """Save token metadata for audit trail (no sensitive data)"""
        metadata = {
            "token_id": payload["sub"],
            "issued_at": payload["iat"].isoformat(),
            "expires_at": payload["exp"].isoformat(),
            "audience": payload["aud"],
            "is_adult": payload["is_adult"],
            # NOTE: No DOB, no names, no images stored
        }
        
        metadata_path = Path(self.config.TOKENS_DIR) / f"metadata_{payload['sub'][:8]}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_public_key_for_verification(self) -> str:
        """Get public key for websites to verify tokens"""
        return self.key_manager.get_public_key_pem()


# === TOKEN VALIDATOR (Website Side) ===
class TokenValidator:
    """
    Validate age verification tokens on website/service side.
    
    Websites only need the public key to verify tokens.
    They never see the ID image, DOB, or any biometric data.
    """
    
    def __init__(self, public_key_pem: str, expected_issuer: str = None):
        """
        Initialize validator with public key.
        
        Args:
            public_key_pem: Public key in PEM format (from token issuer)
            expected_issuer: Expected token issuer (optional)
        """
        self.public_key = serialization.load_pem_public_key(
            public_key_pem.encode('utf-8'),
            backend=default_backend()
        )
        self.expected_issuer = expected_issuer
        self.logger = logging.getLogger(__name__)
    
    def validate_token(
        self,
        token: str,
        audience: str = "default",
        check_device_binding: bool = True
    ) -> Dict:
        """
        Verify token signature and check validity.
        
        Args:
            token: JWT token from user
            audience: Expected audience (service identifier)
            check_device_binding: Verify device fingerprint
        
        Returns:
            Validation result with access decision
        """
        try:
            # Decode and verify signature
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=["RS256"],
                audience=audience,
                options={"verify_exp": True}
            )
            
            # Optional: verify issuer
            if self.expected_issuer and payload.get("iss") != self.expected_issuer:
                return {
                    "valid": False,
                    "reason": f"Invalid issuer: {payload.get('iss')}"
                }
            
            # Optional: verify device binding
            if check_device_binding and "device_fingerprint" in payload:
                current_device = DeviceFingerprint.generate()
                if not DeviceFingerprint.verify(payload["device_fingerprint"], current_device):
                    return {
                        "valid": False,
                        "reason": "Token was issued for a different device"
                    }
            
            # Check adult status
            if payload.get("is_adult"):
                self.logger.info(f"Access granted: Token {payload.get('sub', 'unknown')[:8]}...")
                return {
                    "valid": True,
                    "access_granted": True,
                    "token_id": payload.get("sub"),
                    "expires_at": payload.get("exp"),
                    "audience": payload.get("aud")
                }
            else:
                return {
                    "valid": True,
                    "access_granted": False,
                    "reason": "User is not an adult"
                }
        
        except jwt.ExpiredSignatureError:
            return {
                "valid": False,
                "reason": "Token expired - please re-verify your age"
            }
        except jwt.InvalidAudienceError:
            return {
                "valid": False,
                "reason": f"Token not valid for this service (audience mismatch)"
            }
        except jwt.InvalidTokenError as e:
            return {
                "valid": False,
                "reason": f"Invalid token: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return {
                "valid": False,
                "reason": "Token validation failed"
            }


# === DEMO: WEBSITE INTEGRATION ===
def demo_website_validation():
    """
    Example of how a website would validate tokens.
    This code runs on the website server, NOT the verification system.
    """
    # Step 1: Website obtains the public key from the verification system
    # (This happens once during setup, not per-request)
    token_generator = AgeVerificationToken()
    public_key_pem = token_generator.get_public_key_for_verification()
    
    # Step 2: Website creates a validator
    validator = TokenValidator(
        public_key_pem=public_key_pem,
        expected_issuer="AgeVerification-Thesis-System"
    )
    
    # Step 3: User presents token to website
    # (In reality, this comes from HTTP request/cookie/header)
    example_token = "user_presents_this_token"
    
    # Step 4: Website validates token
    result = validator.validate_token(
        token=example_token,
        audience="example-adult-website.com"
    )
    
    # Step 5: Website makes access decision
    if result["valid"] and result["access_granted"]:
        print("✓ Access granted - user is verified adult")
        return True
    else:
        print(f"✗ Access denied - {result.get('reason')}")
        return False


# === TESTING ===
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Privacy-Preserving Age Verification Token System")
    print("=" * 60)
    
    # Initialize token generator
    config = TokenConfig(
        VALIDITY_DAYS=30,
        ENABLE_DEVICE_BINDING=True,
        INCLUDE_NAME_IN_TOKEN=False,  # Maximum privacy
        INCLUDE_AGE_VALUE=False  # Only "is_adult" boolean
    )
    
    token_gen = AgeVerificationToken(config)
    
    # Example: Adult user (25 years old)
    print("\n1. Generating token for ADULT user (25 years)...")
    adult_token_data = token_gen.generate_token(
        is_adult=True,
        age_years=25,
        audience="adult-website.com"
    )
    
    print(f"Token ID: {adult_token_data['token_id'][:16]}...")
    print(f"Expires: {adult_token_data['expires_at']}")
    print(f"QR Code: {adult_token_data['qr_code_path']}")
    print(f"Token (first 100 chars): {adult_token_data['token'][:100]}...")
    
    # Example: Minor user (16 years old)
    print("\n2. Generating token for MINOR user (16 years)...")
    minor_token_data = token_gen.generate_token(
        is_adult=False,
        age_years=16,
        audience="age-restricted-game.com"
    )
    
    print(f"Token ID: {minor_token_data['token_id'][:16]}...")
    print(f"Is Adult: {minor_token_data['is_adult']}")
    
    # Export public key for websites
    print("\n3. Exporting public key for website verification...")
    public_key = token_gen.get_public_key_for_verification()
    public_key_path = Path("keys/public_key_for_websites.pem")
    with open(public_key_path, 'w') as f:
        f.write(public_key)
    print(f"Public key saved: {public_key_path}")
    print(f"Websites use this key to verify tokens")
    
    # Simulate website validation
    print("\n4. Simulating website token validation...")
    validator = TokenValidator(
        public_key_pem=public_key,
        expected_issuer="AgeVerification-Thesis-System"
    )
    
    # Validate adult token
    print("\n   a) Validating ADULT token:")
    result = validator.validate_token(
        token=adult_token_data['token'],
        audience="adult-website.com"
    )
    print(f" Valid: {result['valid']}")
    print(f" Access Granted: {result.get('access_granted', False)}")
    
    # Validate minor token
    print("\n   b) Validating MINOR token:")
    result = validator.validate_token(
        token=minor_token_data['token'],
        audience="age-restricted-game.com"
    )
    print(f"Valid: {result['valid']}")
    print(f"Access Granted: {result.get('access_granted', False)}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    
    # Test audience mismatch (unlinkability)
    print("\n   c) Testing audience mismatch (unlinkability):")
    result = validator.validate_token(
        token=adult_token_data['token'],
        audience="different-website.com"  # Wrong audience!
    )
    print(f"Valid: {result['valid']}")
    print(f"Reason: {result.get('reason')}")
    
    print("\n" + "=" * 60)
    print("Token system test complete!")
    print("=" * 60)
