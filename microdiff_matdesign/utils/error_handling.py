"""Comprehensive error handling for MicroDiff-MatDesign."""

import traceback
import sys
from typing import Optional, Dict, Any, Callable, Type, Union
from functools import wraps
from contextlib import contextmanager
from enum import Enum

from .logging_config import get_logger, log_exception


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MicroDiffError(Exception):
    """Base exception for MicroDiff-MatDesign errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.context = context or {}
        self.traceback_info = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'error_code': self.error_code,
            'context': self.context,
            'traceback': self.traceback_info
        }


class ValidationError(MicroDiffError):
    """Validation errors."""
    pass


class ProcessingError(MicroDiffError):
    """Processing errors."""
    pass


class ModelError(MicroDiffError):
    """Model-related errors."""
    pass


class DataError(MicroDiffError):
    """Data-related errors."""
    pass


class OptimizationError(MicroDiffError):
    """Optimization errors."""
    pass


class ConfigurationError(MicroDiffError):
    """Configuration errors."""
    pass


class ResourceError(MicroDiffError):
    """Resource-related errors (memory, disk, etc.)."""
    pass


class NetworkError(MicroDiffError):
    """Network-related errors."""
    pass


class ErrorHandler:
    """Central error handling and recovery system."""
    
    def __init__(self):
        self.logger = get_logger('error_handler')
        self.error_count = 0
        self.error_history = []
        self.recovery_strategies = {}
        self.max_history = 1000
    
    def register_recovery_strategy(self, error_type: Type[Exception], 
                                 strategy: Callable[[Exception], Any]) -> None:
        """Register a recovery strategy for an error type.
        
        Args:
            error_type: Type of exception to handle
            strategy: Function to call for recovery
        """
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(self, error: Exception, operation: str = "unknown", 
                    attempt_recovery: bool = True) -> Optional[Any]:
        """Handle an error with logging and optional recovery.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        self.error_count += 1
        
        # Create error record
        error_record = {
            'timestamp': self._get_timestamp(),
            'error_type': type(error).__name__,
            'message': str(error),
            'operation': operation,
            'error_id': self.error_count
        }
        
        # Add to history
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Log the error
        log_exception(self.logger, operation, error)
        
        # Attempt recovery if enabled
        if attempt_recovery:
            recovery_result = self._attempt_recovery(error, operation)
            if recovery_result is not None:
                self.logger.info(f"Successfully recovered from error in {operation}")
                return recovery_result
        
        # Re-raise if no recovery
        self.logger.error(f"No recovery possible for error in {operation}")
        return None
    
    def _attempt_recovery(self, error: Exception, operation: str) -> Optional[Any]:
        """Attempt to recover from an error.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation
            
        Returns:
            Recovery result if successful, None otherwise
        """
        error_type = type(error)
        
        # Check for exact type match
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {error_type.__name__}")
                return self.recovery_strategies[error_type](error)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        # Check for parent type matches
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    self.logger.info(f"Attempting recovery using parent type {registered_type.__name__}")
                    return strategy(error)
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.error_history:
            return {'total_errors': 0}
        
        # Count by error type
        error_types = {}
        for record in self.error_history:
            error_type = record['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Recent errors (last 10)
        recent_errors = self.error_history[-10:]
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recent_errors': recent_errors,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(operation: str = None, reraise: bool = True, 
                 return_on_error: Any = None, attempt_recovery: bool = True):
    """Decorator for automatic error handling.
    
    Args:
        operation: Description of the operation
        reraise: Whether to reraise the exception after handling
        return_on_error: Value to return if error occurs and not reraising
        attempt_recovery: Whether to attempt recovery
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovery_result = error_handler.handle_error(
                    e, op_name, attempt_recovery
                )
                
                if recovery_result is not None:
                    return recovery_result
                
                if reraise:
                    raise
                else:
                    return return_on_error
        
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, reraise: bool = True, 
                 attempt_recovery: bool = True):
    """Context manager for error handling.
    
    Args:
        operation: Description of the operation
        reraise: Whether to reraise exceptions
        attempt_recovery: Whether to attempt recovery
    """
    try:
        yield
    except Exception as e:
        recovery_result = error_handler.handle_error(
            e, operation, attempt_recovery
        )
        
        if recovery_result is None and reraise:
            raise


def validate_input(condition: bool, message: str, error_type: Type[MicroDiffError] = ValidationError,
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                  error_code: Optional[str] = None, 
                  context: Optional[Dict[str, Any]] = None) -> None:
    """Validate input condition and raise error if false.
    
    Args:
        condition: Condition to validate
        message: Error message if condition is false
        error_type: Type of error to raise
        severity: Error severity
        error_code: Optional error code
        context: Optional context information
    """
    if not condition:
        raise error_type(message, severity, error_code, context)


def safe_execute(func: Callable, *args, default_return: Any = None, 
                log_errors: bool = True, **kwargs) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        default_return: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for function
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = get_logger('safe_execute')
            log_exception(logger, f"safe_execute({func.__name__})", e)
        return default_return


def retry_on_error(max_attempts: int = 3, delay: float = 1.0, 
                  backoff_factor: float = 2.0, 
                  exceptions: tuple = (Exception,)):
    """Decorator for retrying operations on error.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Factor to multiply delay by after each failure
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger = get_logger('retry')
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            # If we get here, all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def setup_error_recovery():
    """Setup default error recovery strategies."""
    
    def memory_error_recovery(error: MemoryError) -> None:
        """Recovery strategy for memory errors."""
        import gc
        logger = get_logger('recovery')
        logger.info("Attempting memory cleanup...")
        gc.collect()
        logger.info("Memory cleanup completed")
    
    def file_not_found_recovery(error: FileNotFoundError) -> None:
        """Recovery strategy for file not found errors."""
        logger = get_logger('recovery')
        logger.info("Attempting to create missing directories...")
        
        # Try to create parent directory if it's in the error message
        if hasattr(error, 'filename') and error.filename:
            from pathlib import Path
            path = Path(error.filename)
            if path.parent.exists() == False:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path.parent}")
    
    def permission_error_recovery(error: PermissionError) -> None:
        """Recovery strategy for permission errors."""
        logger = get_logger('recovery')
        logger.warning("Permission denied. Please check file/directory permissions.")
        
        if hasattr(error, 'filename') and error.filename:
            logger.warning(f"Affected file/directory: {error.filename}")
    
    # Register recovery strategies
    error_handler.register_recovery_strategy(MemoryError, memory_error_recovery)
    error_handler.register_recovery_strategy(FileNotFoundError, file_not_found_recovery)
    error_handler.register_recovery_strategy(PermissionError, permission_error_recovery)


def get_error_summary() -> Dict[str, Any]:
    """Get summary of all errors that have occurred.
    
    Returns:
        Dictionary with error summary
    """
    return error_handler.get_error_statistics()


# Initialize error recovery
setup_error_recovery()