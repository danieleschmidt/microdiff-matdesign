"""Security utilities for MicroDiff-MatDesign."""

import hashlib
import secrets
import base64
import os
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

# Try to import cryptography, fall back to basic implementation if not available
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from .logging_config import get_logger
from .error_handling import MicroDiffError, ErrorSeverity


class SecurityError(MicroDiffError):
    """Security-related errors."""
    pass


class InputValidator:
    """Input validation and sanitization utilities."""
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], 
                          allowed_extensions: Optional[List[str]] = None,
                          max_path_length: int = 260) -> Path:
        """Validate and sanitize file paths.
        
        Args:
            path: File path to validate
            allowed_extensions: List of allowed file extensions
            max_path_length: Maximum allowed path length
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is invalid or unsafe
        """
        logger = get_logger('security.validator')
        
        # Convert to Path object
        path_obj = Path(path)
        
        # Check path length
        if len(str(path_obj)) > max_path_length:
            raise SecurityError(
                f"Path too long: {len(str(path_obj))} > {max_path_length}",
                severity=ErrorSeverity.HIGH,
                error_code="PATH_TOO_LONG"
            )
        
        # Check for path traversal attempts
        resolved_path = path_obj.resolve()
        if '..' in str(path_obj) or str(resolved_path) != str(path_obj.absolute()):
            logger.warning(f"Potential path traversal attempt: {path}")
            raise SecurityError(
                "Path traversal detected",
                severity=ErrorSeverity.HIGH,
                error_code="PATH_TRAVERSAL"
            )
        
        # Check file extension if restrictions are specified
        if allowed_extensions:
            extension = path_obj.suffix.lower()
            if extension not in [ext.lower() for ext in allowed_extensions]:
                raise SecurityError(
                    f"File extension '{extension}' not allowed. Allowed: {allowed_extensions}",
                    severity=ErrorSeverity.MEDIUM,
                    error_code="INVALID_EXTENSION"
                )
        
        return path_obj
    
    @staticmethod
    def validate_parameter_value(value: Any, param_name: str, 
                                min_value: Optional[float] = None,
                                max_value: Optional[float] = None,
                                allowed_values: Optional[List[Any]] = None) -> Any:
        """Validate process parameter values.
        
        Args:
            value: Value to validate
            param_name: Name of the parameter
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allowed_values: List of allowed values
            
        Returns:
            Validated value
            
        Raises:
            SecurityError: If value is invalid
        """
        # Check for None values
        if value is None:
            raise SecurityError(
                f"Parameter '{param_name}' cannot be None",
                severity=ErrorSeverity.MEDIUM,
                error_code="NULL_PARAMETER"
            )
        
        # Check allowed values list
        if allowed_values is not None and value not in allowed_values:
            raise SecurityError(
                f"Parameter '{param_name}' value '{value}' not in allowed values: {allowed_values}",
                severity=ErrorSeverity.MEDIUM,
                error_code="INVALID_VALUE"
            )
        
        # Check numeric ranges
        if isinstance(value, (int, float)):
            if min_value is not None and value < min_value:
                raise SecurityError(
                    f"Parameter '{param_name}' value {value} below minimum {min_value}",
                    severity=ErrorSeverity.MEDIUM,
                    error_code="VALUE_TOO_LOW"
                )
            
            if max_value is not None and value > max_value:
                raise SecurityError(
                    f"Parameter '{param_name}' value {value} above maximum {max_value}",
                    severity=ErrorSeverity.MEDIUM,
                    error_code="VALUE_TOO_HIGH"
                )
        
        return value
    
    @staticmethod
    def sanitize_string(input_string: str, max_length: int = 1000) -> str:
        """Sanitize string input.
        
        Args:
            input_string: String to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If string is invalid
        """
        if not isinstance(input_string, str):
            raise SecurityError(
                "Input must be a string",
                severity=ErrorSeverity.MEDIUM,
                error_code="INVALID_TYPE"
            )
        
        # Check length
        if len(input_string) > max_length:
            raise SecurityError(
                f"String too long: {len(input_string)} > {max_length}",
                severity=ErrorSeverity.MEDIUM,
                error_code="STRING_TOO_LONG"
            )
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '`', '\\', '|', ';']
        sanitized = input_string
        
        for char in dangerous_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()


class SecureStorage:
    """Secure storage for sensitive data."""
    
    def __init__(self, password: Optional[str] = None):
        """Initialize secure storage.
        
        Args:
            password: Password for encryption (if None, generates random key)
        """
        self.logger = get_logger('security.storage')
        
        if not HAS_CRYPTOGRAPHY:
            self.logger.warning("Cryptography library not available, using basic encoding")
            self.cipher = None
            self._password = password or "default_key"
        else:
            if password:
                # Derive key from password
                salt = b'microdiff_salt_v1'  # In production, use random salt
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            else:
                # Generate random key
                key = Fernet.generate_key()
            
            self.cipher = Fernet(key)
        
        self.logger.info("Secure storage initialized")
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if self.cipher:
            return self.cipher.encrypt(data)
        else:
            # Fallback: simple base64 encoding (NOT secure!)
            return base64.b64encode(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data
            
        Raises:
            SecurityError: If decryption fails
        """
        try:
            if self.cipher:
                return self.cipher.decrypt(encrypted_data)
            else:
                # Fallback: base64 decoding
                return base64.b64decode(encrypted_data)
        except Exception as e:
            raise SecurityError(
                f"Decryption failed: {e}",
                severity=ErrorSeverity.HIGH,
                error_code="DECRYPTION_FAILED"
            )
    
    def store_secure_config(self, config_data: Dict[str, Any], 
                           file_path: Union[str, Path]) -> None:
        """Store configuration data securely.
        
        Args:
            config_data: Configuration data to store
            file_path: Path to store encrypted file
        """
        import json
        
        # Serialize config
        json_data = json.dumps(config_data, indent=2)
        
        # Encrypt
        encrypted_data = self.encrypt_data(json_data)
        
        # Store
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
        
        # Set secure permissions
        os.chmod(file_path, 0o600)  # Read/write for owner only
        
        self.logger.info(f"Stored secure config to {file_path}")
    
    def load_secure_config(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load secure configuration data.
        
        Args:
            file_path: Path to encrypted file
            
        Returns:
            Configuration data
            
        Raises:
            SecurityError: If loading fails
        """
        import json
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise SecurityError(
                f"Secure config file not found: {file_path}",
                severity=ErrorSeverity.HIGH,
                error_code="FILE_NOT_FOUND"
            )
        
        # Check file permissions
        stat_info = file_path.stat()
        if stat_info.st_mode & 0o077:  # Check if others have access
            self.logger.warning(f"Insecure permissions on {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted_data = self.decrypt_data(encrypted_data)
            
            # Parse JSON
            config_data = json.loads(decrypted_data.decode('utf-8'))
            
            self.logger.info(f"Loaded secure config from {file_path}")
            return config_data
            
        except Exception as e:
            raise SecurityError(
                f"Failed to load secure config: {e}",
                severity=ErrorSeverity.HIGH,
                error_code="CONFIG_LOAD_FAILED"
            )


class AccessControl:
    """Access control and permission management."""
    
    def __init__(self):
        self.logger = get_logger('security.access_control')
        self.permissions: Dict[str, List[str]] = {}
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
    
    def add_permission(self, user: str, permission: str) -> None:
        """Add permission for user.
        
        Args:
            user: User identifier
            permission: Permission to add
        """
        if user not in self.permissions:
            self.permissions[user] = []
        
        if permission not in self.permissions[user]:
            self.permissions[user].append(permission)
            self.logger.info(f"Added permission '{permission}' for user '{user}'")
    
    def remove_permission(self, user: str, permission: str) -> None:
        """Remove permission for user.
        
        Args:
            user: User identifier
            permission: Permission to remove
        """
        if user in self.permissions and permission in self.permissions[user]:
            self.permissions[user].remove(permission)
            self.logger.info(f"Removed permission '{permission}' for user '{user}'")
    
    def has_permission(self, user: str, permission: str) -> bool:
        """Check if user has permission.
        
        Args:
            user: User identifier
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        return user in self.permissions and permission in self.permissions[user]
    
    def create_session_token(self, user: str, permissions: List[str]) -> str:
        """Create session token for user.
        
        Args:
            user: User identifier
            permissions: List of permissions for this session
            
        Returns:
            Session token
        """
        token = secrets.token_urlsafe(32)
        
        self.session_tokens[token] = {
            'user': user,
            'permissions': permissions,
            'created_at': time.time(),
            'expires_at': time.time() + 3600  # 1 hour
        }
        
        self.logger.info(f"Created session token for user '{user}'")
        return token
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate session token.
        
        Args:
            token: Session token to validate
            
        Returns:
            Session data if valid, None otherwise
        """
        if token not in self.session_tokens:
            return None
        
        session = self.session_tokens[token]
        
        # Check expiration
        if time.time() > session['expires_at']:
            del self.session_tokens[token]
            self.logger.warning(f"Expired session token removed")
            return None
        
        return session
    
    def revoke_session_token(self, token: str) -> None:
        """Revoke session token.
        
        Args:
            token: Session token to revoke
        """
        if token in self.session_tokens:
            del self.session_tokens[token]
            self.logger.info("Session token revoked")


class SecurityAuditor:
    """Security audit and vulnerability scanning."""
    
    def __init__(self):
        self.logger = get_logger('security.auditor')
        self.vulnerabilities: List[Dict[str, Any]] = []
    
    def audit_file_permissions(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Audit file permissions.
        
        Args:
            file_path: Path to audit
            
        Returns:
            List of security issues found
        """
        issues = []
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            return issues
        
        stat_info = path_obj.stat()
        mode = stat_info.st_mode
        
        # Check for world-writable files
        if mode & 0o002:
            issues.append({
                'type': 'file_permissions',
                'severity': 'high',
                'file': str(path_obj),
                'issue': 'File is world-writable',
                'recommendation': 'Remove write permission for others'
            })
        
        # Check for world-readable sensitive files
        if path_obj.suffix in ['.key', '.pem', '.p12', '.pfx'] and mode & 0o004:
            issues.append({
                'type': 'file_permissions',
                'severity': 'medium',
                'file': str(path_obj),
                'issue': 'Sensitive file is world-readable',
                'recommendation': 'Restrict read access to owner only'
            })
        
        return issues
    
    def audit_directory(self, directory_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Audit directory for security issues.
        
        Args:
            directory_path: Directory to audit
            
        Returns:
            List of security issues found
        """
        issues = []
        dir_path = Path(directory_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            return issues
        
        # Recursively check files
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                file_issues = self.audit_file_permissions(file_path)
                issues.extend(file_issues)
        
        return issues
    
    def scan_for_secrets(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Scan file for potential secrets.
        
        Args:
            file_path: File to scan
            
        Returns:
            List of potential secrets found
        """
        secrets_found = []
        path_obj = Path(file_path)
        
        if not path_obj.exists() or not path_obj.is_file():
            return secrets_found
        
        # Patterns that might indicate secrets
        secret_patterns = [
            r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^"\'\s]+)',
            r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([^"\'\s]+)',
            r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?([^"\'\s]+)',
            r'(?i)(token)\s*[=:]\s*["\']?([^"\'\s]{20,})',
            r'["\']?[A-Za-z0-9]{32,}["\']?',  # Long hex strings
        ]
        
        try:
            with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            import re
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in secret_patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        secrets_found.append({
                            'type': 'potential_secret',
                            'severity': 'high',
                            'file': str(path_obj),
                            'line': i,
                            'pattern': pattern,
                            'match': match.group(0)[:50] + '...' if len(match.group(0)) > 50 else match.group(0)
                        })
        
        except Exception as e:
            self.logger.warning(f"Could not scan {path_obj} for secrets: {e}")
        
        return secrets_found
    
    def generate_security_report(self, scan_path: Union[str, Path]) -> Dict[str, Any]:
        """Generate comprehensive security report.
        
        Args:
            scan_path: Path to scan (file or directory)
            
        Returns:
            Security report
        """
        scan_path = Path(scan_path)
        report = {
            'scan_path': str(scan_path),
            'scan_time': time.time(),
            'issues': [],
            'summary': {
                'total_issues': 0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0
            }
        }
        
        # Audit permissions
        if scan_path.is_dir():
            permission_issues = self.audit_directory(scan_path)
            report['issues'].extend(permission_issues)
            
            # Scan for secrets in all files
            for file_path in scan_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in ['.py', '.txt', '.json', '.yaml', '.yml', '.conf']:
                    secret_issues = self.scan_for_secrets(file_path)
                    report['issues'].extend(secret_issues)
        else:
            permission_issues = self.audit_file_permissions(scan_path)
            secret_issues = self.scan_for_secrets(scan_path)
            report['issues'].extend(permission_issues + secret_issues)
        
        # Calculate summary
        for issue in report['issues']:
            severity = issue.get('severity', 'low')
            report['summary'][f'{severity}_severity'] += 1
        
        report['summary']['total_issues'] = len(report['issues'])
        
        return report


# Global security components
input_validator = InputValidator()
access_control = AccessControl()
security_auditor = SecurityAuditor()


def require_permission(permission: str):
    """Decorator to require specific permission.
    
    Args:
        permission: Required permission
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, this would check the current user's permissions
            # For now, we'll just log the requirement
            logger = get_logger('security.permission_check')
            logger.debug(f"Function {func.__name__} requires permission: {permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def secure_temp_file(suffix: str = '.tmp', prefix: str = 'microdiff_') -> Path:
    """Create a secure temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        
    Returns:
        Path to secure temporary file
    """
    import tempfile
    
    # Create temporary file with secure permissions
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    
    # Set secure permissions (owner read/write only)
    os.chmod(temp_path, 0o600)
    
    return Path(temp_path)


def hash_data(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """Hash data securely.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hexadecimal hash digest
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if algorithm == 'sha256':
        return hashlib.sha256(data).hexdigest()
    elif algorithm == 'sha512':
        return hashlib.sha512(data).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(data).hexdigest()
    else:
        raise SecurityError(f"Unsupported hash algorithm: {algorithm}")


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        URL-safe base64 encoded token
    """
    return secrets.token_urlsafe(length)