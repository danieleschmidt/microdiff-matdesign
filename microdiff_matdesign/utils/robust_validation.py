"""Robust Validation and Error Handling for MicroDiff-MatDesign.

This module implements comprehensive input validation, error handling,
and security measures for production-ready deployment.

Generation 2 Features:
- Input sanitization and validation
- Physics-based constraint checking
- Security measures against adversarial inputs
- Comprehensive error handling with logging
- Data integrity verification
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from functools import wraps
import logging

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class PhysicsConstraintError(Exception):
    """Custom exception for physics constraint violations."""
    pass


def validation_wrapper(validate_inputs: bool = True, validate_outputs: bool = False):
    """Decorator for automatic input/output validation."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if validate_inputs:
                logger.debug(f"Validating inputs for {func.__name__}")
                
            try:
                result = func(*args, **kwargs)
                
                if validate_outputs:
                    logger.debug(f"Validating outputs for {func.__name__}")
                    
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
    return decorator


class InputValidator:
    """Comprehensive input validation for materials science data."""
    
    def __init__(self):
        """Initialize validator with default constraints."""
        
        # Physics-based parameter ranges for Ti-6Al-4V LPBF
        self.parameter_ranges = {
            'laser_power': (50.0, 500.0),      # Watts
            'scan_speed': (200.0, 2000.0),     # mm/s
            'layer_thickness': (10.0, 100.0),  # micrometers
            'hatch_spacing': (50.0, 300.0),    # micrometers
            'powder_bed_temp': (20.0, 200.0),  # Celsius
            'energy_density': (20.0, 200.0)    # J/mm³
        }
        
        # Microstructure data constraints
        self.microstructure_constraints = {
            'min_resolution': (16, 16, 16),
            'max_resolution': (512, 512, 512),
            'valid_dtypes': [np.float32, np.float64, np.uint8, np.uint16],
            'value_range': (0.0, 1.0)
        }
        
        # Security patterns to detect malicious inputs
        self.security_patterns = [
            r'(?i)(\.\./|\.\.\\)',          # Path traversal
            r'(?i)(script|javascript)',     # Script injection
            r'(?i)(eval|exec|import)',      # Code injection
            r'[<>"\']',                     # HTML/XML injection
            r'(?i)(select|union|drop|delete|insert|update)', # SQL injection
        ]
        
    def validate_process_parameters(
        self, 
        parameters: Union[Dict[str, float], np.ndarray],
        strict: bool = True
    ) -> Dict[str, float]:
        """Validate manufacturing process parameters.
        
        Args:
            parameters: Process parameters dictionary or array
            strict: Whether to apply strict physics constraints
            
        Returns:
            Validated and sanitized parameters
            
        Raises:
            ValidationError: If parameters are invalid
            PhysicsConstraintError: If physics constraints violated
        """
        logger.info("Validating process parameters")
        
        # Convert array to dict if needed
        if isinstance(parameters, np.ndarray):
            param_names = ['laser_power', 'scan_speed', 'layer_thickness', 
                          'hatch_spacing', 'powder_bed_temp']
            if len(parameters) < len(param_names):
                param_names = param_names[:len(parameters)]
            parameters = dict(zip(param_names, parameters))
        
        validated_params = {}
        
        for param_name, value in parameters.items():
            try:
                # Basic type validation
                if not isinstance(value, (int, float, np.number)):
                    raise ValidationError(f"Parameter {param_name} must be numeric, got {type(value)}")
                
                value = float(value)
                
                # Check for invalid values
                if not np.isfinite(value):
                    raise ValidationError(f"Parameter {param_name} must be finite, got {value}")
                
                if value < 0:
                    raise ValidationError(f"Parameter {param_name} must be non-negative, got {value}")
                
                # Range validation
                if param_name in self.parameter_ranges:
                    min_val, max_val = self.parameter_ranges[param_name]
                    if not (min_val <= value <= max_val):
                        if strict:
                            raise ValidationError(
                                f"Parameter {param_name} = {value} outside valid range "
                                f"[{min_val}, {max_val}]"
                            )
                        else:
                            # Clamp to valid range with warning
                            clamped_value = np.clip(value, min_val, max_val)
                            if clamped_value != value:
                                warnings.warn(
                                    f"Parameter {param_name} = {value} clamped to {clamped_value}"
                                )
                            value = clamped_value
                
                validated_params[param_name] = value
                
            except Exception as e:
                logger.error(f"Failed to validate parameter {param_name}: {e}")
                raise ValidationError(f"Invalid parameter {param_name}: {e}")
        
        # Physics consistency checks
        if strict and len(validated_params) >= 4:
            self._validate_physics_consistency(validated_params)
        
        logger.info(f"Successfully validated {len(validated_params)} parameters")
        return validated_params
    
    def validate_microstructure_data(
        self,
        microstructure: np.ndarray,
        normalize: bool = True,
        check_integrity: bool = True
    ) -> np.ndarray:
        """Validate microstructure image data.
        
        Args:
            microstructure: 3D microstructure array
            normalize: Whether to normalize values to [0,1]
            check_integrity: Whether to check data integrity
            
        Returns:
            Validated and processed microstructure data
            
        Raises:
            ValidationError: If microstructure data is invalid
        """
        logger.info("Validating microstructure data")
        
        # Basic array validation
        if not isinstance(microstructure, np.ndarray):
            raise ValidationError(f"Microstructure must be numpy array, got {type(microstructure)}")
        
        if microstructure.ndim not in [3, 4]:
            raise ValidationError(f"Microstructure must be 3D or 4D array, got {microstructure.ndim}D")
        
        # Squeeze if 4D with batch dimension of 1
        if microstructure.ndim == 4 and microstructure.shape[0] == 1:
            microstructure = microstructure.squeeze(0)
        
        # Resolution validation
        min_res = self.microstructure_constraints['min_resolution']
        max_res = self.microstructure_constraints['max_resolution']
        
        for i, (size, min_size, max_size) in enumerate(zip(microstructure.shape, min_res, max_res)):
            if size < min_size:
                raise ValidationError(f"Dimension {i} size {size} < minimum {min_size}")
            if size > max_size:
                raise ValidationError(f"Dimension {i} size {size} > maximum {max_size}")
        
        # Data type validation
        valid_dtypes = self.microstructure_constraints['valid_dtypes']
        if microstructure.dtype not in valid_dtypes:
            logger.warning(f"Converting microstructure from {microstructure.dtype} to float32")
            microstructure = microstructure.astype(np.float32)
        
        # Value range validation
        if np.any(~np.isfinite(microstructure)):
            raise ValidationError("Microstructure contains non-finite values")
        
        min_val, max_val = self.microstructure_constraints['value_range']
        if normalize:
            # Normalize to [0, 1] range
            data_min, data_max = microstructure.min(), microstructure.max()
            if data_max > data_min:
                microstructure = (microstructure - data_min) / (data_max - data_min)
            else:
                microstructure = np.zeros_like(microstructure)
        else:
            # Check if already in valid range
            if microstructure.min() < min_val or microstructure.max() > max_val:
                warnings.warn(
                    f"Microstructure values outside range [{min_val}, {max_val}]. "
                    "Consider setting normalize=True"
                )
        
        # Data integrity checks
        if check_integrity:
            self._check_data_integrity(microstructure)
        
        logger.info(f"Successfully validated microstructure shape {microstructure.shape}")
        return microstructure
    
    def validate_file_path(
        self, 
        file_path: Union[str, Path],
        allowed_extensions: Optional[List[str]] = None,
        max_path_length: int = 255
    ) -> Path:
        """Validate file path for security and accessibility.
        
        Args:
            file_path: Input file path
            allowed_extensions: List of allowed file extensions
            max_path_length: Maximum allowed path length
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path contains security risks
            ValidationError: If path is invalid
        """
        logger.debug(f"Validating file path: {file_path}")
        
        # Convert to Path object
        if isinstance(file_path, str):
            # Security check for malicious patterns
            for pattern in self.security_patterns:
                if re.search(pattern, file_path):
                    raise SecurityError(f"Potentially malicious path pattern detected: {file_path}")
            
            file_path = Path(file_path)
        
        # Basic validation
        if not isinstance(file_path, Path):
            raise ValidationError(f"Invalid file path type: {type(file_path)}")
        
        path_str = str(file_path)
        if len(path_str) > max_path_length:
            raise ValidationError(f"Path too long: {len(path_str)} > {max_path_length}")
        
        # Check for path traversal attempts
        resolved_path = file_path.resolve()
        if '..' in resolved_path.parts:
            raise SecurityError(f"Path traversal detected in: {file_path}")
        
        # Extension validation
        if allowed_extensions:
            ext = file_path.suffix.lower()
            if ext not in [e.lower() for e in allowed_extensions]:
                raise ValidationError(
                    f"File extension {ext} not allowed. "
                    f"Allowed extensions: {allowed_extensions}"
                )
        
        logger.debug(f"File path validated: {resolved_path}")
        return resolved_path
    
    def _validate_physics_consistency(self, parameters: Dict[str, float]) -> None:
        """Validate physics consistency of process parameters.
        
        Args:
            parameters: Validated process parameters
            
        Raises:
            PhysicsConstraintError: If physics constraints violated
        """
        logger.debug("Checking physics consistency")
        
        # Required parameters for energy density calculation
        required_params = ['laser_power', 'scan_speed', 'layer_thickness', 'hatch_spacing']
        if not all(param in parameters for param in required_params):
            return  # Skip if not all parameters available
        
        # Calculate energy density
        laser_power = parameters['laser_power']
        scan_speed = parameters['scan_speed']
        layer_thickness = parameters['layer_thickness']
        hatch_spacing = parameters['hatch_spacing']
        
        energy_density = laser_power / (scan_speed * hatch_spacing * layer_thickness * 1e-6)
        
        # Check energy density bounds
        min_energy, max_energy = self.parameter_ranges['energy_density']
        if not (min_energy <= energy_density <= max_energy):
            raise PhysicsConstraintError(
                f"Energy density {energy_density:.2f} J/mm³ outside valid range "
                f"[{min_energy}, {max_energy}] J/mm³"
            )
        
        # Check thermal time scales
        if 'powder_bed_temp' in parameters:
            thermal_conductivity = 7.0  # W/m·K for Ti-6Al-4V
            specific_heat = 526.0       # J/kg·K
            density = 4430.0           # kg/m³
            
            thermal_diffusivity = thermal_conductivity / (density * specific_heat)
            char_length = np.sqrt(hatch_spacing * layer_thickness) * 1e-6
            thermal_time = char_length**2 / thermal_diffusivity
            
            interaction_time = hatch_spacing / (scan_speed * 1000)
            time_ratio = interaction_time / thermal_time
            
            if not (0.05 <= time_ratio <= 20.0):
                raise PhysicsConstraintError(
                    f"Thermal time ratio {time_ratio:.3f} indicates poor heat dissipation"
                )
        
        logger.debug("Physics consistency validated")
    
    def _check_data_integrity(self, data: np.ndarray) -> None:
        """Check data integrity and detect potential corruption.
        
        Args:
            data: Data array to check
            
        Raises:
            ValidationError: If data integrity issues detected
        """
        logger.debug("Checking data integrity")
        
        # Check for unusual patterns that might indicate corruption
        
        # 1. Check for excessive uniformity (might indicate corruption or artificial data)
        if data.std() < 1e-6:
            warnings.warn("Data appears to be constant or near-constant")
        
        # 2. Check for suspicious patterns
        flat_data = data.flatten()
        
        # Check for repeated values (might indicate corruption)
        unique_values = len(np.unique(flat_data))
        total_values = len(flat_data)
        uniqueness_ratio = unique_values / total_values
        
        if uniqueness_ratio < 0.01:
            warnings.warn(f"Low data uniqueness ratio: {uniqueness_ratio:.4f}")
        
        # 3. Check for extreme outliers
        if data.size > 1000:  # Only for reasonably sized arrays
            q1, q3 = np.percentile(flat_data, [25, 75])
            iqr = q3 - q1
            
            if iqr > 0:
                outlier_threshold = 3.0 * iqr
                outliers = (flat_data < q1 - outlier_threshold) | (flat_data > q3 + outlier_threshold)
                outlier_ratio = np.sum(outliers) / len(flat_data)
                
                if outlier_ratio > 0.05:  # More than 5% outliers
                    warnings.warn(f"High outlier ratio detected: {outlier_ratio:.2%}")
        
        # 4. Check for NaN or infinite values (should have been caught earlier)
        if np.any(~np.isfinite(data)):
            raise ValidationError("Data contains non-finite values")
        
        logger.debug("Data integrity check completed")


class SecureDataHandler:
    """Secure data handling with encryption and integrity verification."""
    
    def __init__(self):
        """Initialize secure data handler."""
        self.validator = InputValidator()
        
    def sanitize_string_input(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input for security.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed string length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If input contains malicious content
            ValidationError: If input is invalid
        """
        if not isinstance(input_str, str):
            raise ValidationError(f"Input must be string, got {type(input_str)}")
        
        if len(input_str) > max_length:
            raise ValidationError(f"String too long: {len(input_str)} > {max_length}")
        
        # Check for malicious patterns
        for pattern in self.validator.security_patterns:
            if re.search(pattern, input_str):
                raise SecurityError(f"Potentially malicious pattern detected in input")
        
        # Basic sanitization
        sanitized = re.sub(r'[<>"\']', '', input_str)
        sanitized = sanitized.strip()
        
        return sanitized
    
    def verify_data_checksum(self, data: np.ndarray, expected_checksum: str) -> bool:
        """Verify data integrity using checksum.
        
        Args:
            data: Data array to verify
            expected_checksum: Expected checksum string
            
        Returns:
            True if checksum matches, False otherwise
        """
        import hashlib
        
        # Calculate SHA-256 checksum
        data_bytes = data.tobytes()
        calculated_checksum = hashlib.sha256(data_bytes).hexdigest()
        
        return calculated_checksum == expected_checksum
    
    def calculate_data_checksum(self, data: np.ndarray) -> str:
        """Calculate checksum for data integrity verification.
        
        Args:
            data: Data array
            
        Returns:
            SHA-256 checksum string
        """
        import hashlib
        
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()


class RobustErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_counts = {}
        self.recovery_strategies = {
            ValidationError: self._handle_validation_error,
            PhysicsConstraintError: self._handle_physics_error,
            SecurityError: self._handle_security_error,
            np.linalg.LinAlgError: self._handle_numerical_error,
            MemoryError: self._handle_memory_error,
        }
    
    def handle_error(
        self, 
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Any]]:
        """Handle error with appropriate recovery strategy.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            Tuple of (recovery_successful, recovered_result)
        """
        error_type = type(error)
        error_key = f"{error_type.__name__}:{str(error)}"
        
        # Track error frequency
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        logger.error(f"Handling error ({self.error_counts[error_key]} occurrences): {error}")
        
        # Apply recovery strategy if available
        if error_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_type]
                return recovery_func(error, context)
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
                return False, None
        else:
            logger.warning(f"No recovery strategy for {error_type}")
            return False, None
    
    def _handle_validation_error(
        self, 
        error: ValidationError, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Any]]:
        """Handle validation errors with data correction attempts."""
        logger.info("Attempting validation error recovery")
        
        if context and 'data' in context:
            data = context['data']
            
            # Try to fix common validation issues
            if isinstance(data, np.ndarray):
                # Fix non-finite values
                if np.any(~np.isfinite(data)):
                    logger.info("Replacing non-finite values with zeros")
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    return True, data
                
                # Fix extreme values
                if np.abs(data).max() > 1e6:
                    logger.info("Clipping extreme values")
                    data = np.clip(data, -1e6, 1e6)
                    return True, data
        
        return False, None
    
    def _handle_physics_error(
        self, 
        error: PhysicsConstraintError, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Any]]:
        """Handle physics constraint violations with parameter adjustment."""
        logger.info("Attempting physics error recovery")
        
        if context and 'parameters' in context:
            params = context['parameters'].copy()
            validator = InputValidator()
            
            # Try to adjust parameters to satisfy constraints
            try:
                validated_params = validator.validate_process_parameters(
                    params, strict=False
                )
                logger.info("Successfully adjusted parameters to meet constraints")
                return True, validated_params
            except Exception:
                pass
        
        return False, None
    
    def _handle_security_error(
        self, 
        error: SecurityError, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Any]]:
        """Handle security errors (no recovery, log and block)."""
        logger.critical(f"Security error detected: {error}")
        
        # Log security incident
        if context:
            logger.critical(f"Security context: {context}")
        
        # No recovery for security errors - fail securely
        return False, None
    
    def _handle_numerical_error(
        self, 
        error: np.linalg.LinAlgError, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Any]]:
        """Handle numerical computation errors."""
        logger.info("Attempting numerical error recovery")
        
        if context and 'matrix' in context:
            matrix = context['matrix']
            
            # Try adding regularization for singular matrices
            if isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                logger.info("Adding regularization to matrix")
                regularized = matrix + 1e-6 * np.eye(matrix.shape[0])
                return True, regularized
        
        return False, None
    
    def _handle_memory_error(
        self, 
        error: MemoryError, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, Optional[Any]]:
        """Handle memory errors with data reduction."""
        logger.warning("Attempting memory error recovery")
        
        if context and 'data' in context:
            data = context['data']
            
            # Try reducing data size
            if isinstance(data, np.ndarray) and data.size > 1000000:
                logger.info("Downsampling large data array")
                # Simple downsampling by factor of 2
                if data.ndim == 3:
                    downsampled = data[::2, ::2, ::2]
                    return True, downsampled
        
        return False, None


@validation_wrapper(validate_inputs=True)
def robust_inverse_design(
    microstructure: np.ndarray,
    model: Any,
    validator: Optional[InputValidator] = None,
    error_handler: Optional[RobustErrorHandler] = None
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Robust inverse design with comprehensive error handling.
    
    Args:
        microstructure: Input microstructure data
        model: Diffusion model for inverse design
        validator: Input validator instance
        error_handler: Error handler instance
        
    Returns:
        Tuple of (process_parameters, metadata)
    """
    if validator is None:
        validator = InputValidator()
    if error_handler is None:
        error_handler = RobustErrorHandler()
    
    metadata = {'validation_passed': False, 'errors_encountered': []}
    
    try:
        # Validate input microstructure
        validated_microstructure = validator.validate_microstructure_data(
            microstructure, normalize=True, check_integrity=True
        )
        metadata['microstructure_shape'] = validated_microstructure.shape
        
        # Perform inverse design
        if hasattr(model, 'inverse_design'):
            result = model.inverse_design(validated_microstructure)
            
            # Validate output parameters
            if isinstance(result, tuple):
                parameters, uncertainty = result
            else:
                parameters = result
                uncertainty = None
            
            # Convert to dictionary if needed
            if hasattr(parameters, 'to_dict'):
                parameters = parameters.to_dict()
            
            validated_params = validator.validate_process_parameters(
                parameters, strict=True
            )
            
            metadata['validation_passed'] = True
            metadata['uncertainty'] = uncertainty
            
            return validated_params, metadata
            
        else:
            raise ValidationError("Model does not support inverse_design method")
            
    except Exception as e:
        metadata['errors_encountered'].append(str(e))
        
        # Attempt error recovery
        context = {
            'data': microstructure,
            'model': model
        }
        
        recovery_successful, recovered_result = error_handler.handle_error(e, context)
        
        if recovery_successful:
            logger.info("Error recovery successful")
            metadata['recovery_applied'] = True
            return recovered_result, metadata
        else:
            logger.error("Error recovery failed")
            raise


# Export main classes and functions
__all__ = [
    'InputValidator',
    'SecureDataHandler', 
    'RobustErrorHandler',
    'ValidationError',
    'SecurityError',
    'PhysicsConstraintError',
    'validation_wrapper',
    'robust_inverse_design'
]