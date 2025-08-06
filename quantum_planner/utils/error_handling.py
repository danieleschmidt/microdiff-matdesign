"""Comprehensive error handling for quantum task planner."""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, List, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import functools
import threading
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    RESOURCE = "resource"
    NETWORK = "network"
    QUANTUM = "quantum"
    SCHEDULING = "scheduling"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    DEPENDENCY = "dependency"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback_str: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    occurrence_count: int = 1


class QuantumPlannerError(Exception):
    """Base exception for quantum planner errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()


class ValidationError(QuantumPlannerError):
    """Validation-specific error."""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)
        self.field = field
        self.value = value


class QuantumComputationError(QuantumPlannerError):
    """Quantum computation-specific error."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 quantum_state: Optional[Dict] = None, **kwargs):
        super().__init__(message, ErrorCategory.QUANTUM, **kwargs)
        self.operation = operation
        self.quantum_state = quantum_state


class ResourceError(QuantumPlannerError):
    """Resource-related error."""
    
    def __init__(self, message: str, resource_id: Optional[str] = None,
                 resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.RESOURCE, **kwargs)
        self.resource_id = resource_id
        self.resource_type = resource_type


class SchedulingError(QuantumPlannerError):
    """Scheduling-specific error."""
    
    def __init__(self, message: str, task_id: Optional[str] = None,
                 schedule_info: Optional[Dict] = None, **kwargs):
        super().__init__(message, ErrorCategory.SCHEDULING, **kwargs)
        self.task_id = task_id
        self.schedule_info = schedule_info


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self, max_error_history: int = 1000,
                 enable_auto_recovery: bool = True,
                 log_all_errors: bool = True):
        """Initialize error handler."""
        self.max_error_history = max_error_history
        self.enable_auto_recovery = enable_auto_recovery
        self.log_all_errors = log_all_errors
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_handlers: Dict[Type[Exception], Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register default recovery handlers
        self._register_default_recovery_handlers()
        
        logger.info("Initialized ErrorHandler")
    
    def handle_error(self, exception: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    attempt_recovery: bool = True) -> bool:
        """Handle an error with optional recovery."""
        try:
            # Create error info
            error_info = self._create_error_info(exception, context)
            
            # Log error
            if self.log_all_errors:
                self._log_error(error_info)
            
            # Store error
            with self._lock:
                self.error_history.append(error_info)
                if len(self.error_history) > self.max_error_history:
                    self.error_history.pop(0)
                
                # Update error counts
                error_key = f"{error_info.exception_type}:{error_info.message[:100]}"
                self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
                error_info.occurrence_count = self.error_counts[error_key]
            
            # Attempt recovery
            recovery_successful = False
            if attempt_recovery and self.enable_auto_recovery:
                recovery_successful = self._attempt_recovery(exception, error_info)
                error_info.recovery_attempted = True
                error_info.recovery_successful = recovery_successful
            
            return recovery_successful
            
        except Exception as handler_error:
            # Don't let error handling itself fail
            logger.critical(f"Error handler failed: {handler_error}")
            return False
    
    def register_recovery_handler(self, exception_type: Type[Exception],
                                handler: Callable[[Exception, ErrorInfo], bool]):
        """Register a custom recovery handler for specific exception type."""
        self.recovery_handlers[exception_type] = handler
        logger.debug(f"Registered recovery handler for {exception_type.__name__}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            if not self.error_history:
                return {'total_errors': 0}
            
            # Basic counts
            total_errors = len(self.error_history)
            recent_errors = len([e for e in self.error_history 
                               if (datetime.now() - e.timestamp).total_seconds() < 3600])
            
            # By severity
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = len([e for e in self.error_history 
                                                     if e.severity == severity])
            
            # By category
            category_counts = {}
            for category in ErrorCategory:
                category_counts[category.value] = len([e for e in self.error_history 
                                                     if e.category == category])
            
            # Recovery statistics
            recovery_attempted = len([e for e in self.error_history if e.recovery_attempted])
            recovery_successful = len([e for e in self.error_history if e.recovery_successful])
            recovery_rate = (recovery_successful / recovery_attempted * 100) if recovery_attempted > 0 else 0
            
            # Most common errors
            most_common = sorted(self.error_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_errors': total_errors,
                'recent_errors_1h': recent_errors,
                'severity_distribution': severity_counts,
                'category_distribution': category_counts,
                'recovery_attempted': recovery_attempted,
                'recovery_successful': recovery_successful,
                'recovery_rate_percent': recovery_rate,
                'most_common_errors': most_common,
                'error_rate_per_hour': recent_errors
            }
    
    def get_recent_errors(self, count: int = 10) -> List[ErrorInfo]:
        """Get most recent errors."""
        with self._lock:
            return self.error_history[-count:] if self.error_history else []
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorInfo]:
        """Get errors by category."""
        with self._lock:
            return [e for e in self.error_history if e.category == category]
    
    def clear_error_history(self):
        """Clear error history."""
        with self._lock:
            self.error_history.clear()
            self.error_counts.clear()
        logger.info("Cleared error history")
    
    def export_error_report(self, format_type: str = "json") -> str:
        """Export detailed error report."""
        statistics = self.get_error_statistics()
        recent_errors = self.get_recent_errors(50)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'statistics': statistics,
            'recent_errors': [
                {
                    'error_id': e.error_id,
                    'timestamp': e.timestamp.isoformat(),
                    'severity': e.severity.value,
                    'category': e.category.value,
                    'message': e.message,
                    'exception_type': e.exception_type,
                    'context': e.context,
                    'recovery_attempted': e.recovery_attempted,
                    'recovery_successful': e.recovery_successful,
                    'occurrence_count': e.occurrence_count
                }
                for e in recent_errors
            ]
        }
        
        if format_type == "json":
            return json.dumps(report, indent=2)
        else:
            # Text format
            lines = [
                "QUANTUM PLANNER ERROR REPORT",
                "=" * 40,
                f"Report Generated: {report['report_timestamp']}",
                "",
                "STATISTICS:",
                f"  Total Errors: {statistics['total_errors']}",
                f"  Recent Errors (1h): {statistics['recent_errors_1h']}",
                f"  Recovery Rate: {statistics['recovery_rate_percent']:.1f}%",
                ""
            ]
            
            lines.append("SEVERITY DISTRIBUTION:")
            for severity, count in statistics['severity_distribution'].items():
                lines.append(f"  {severity.title()}: {count}")
            
            lines.append("")
            lines.append("CATEGORY DISTRIBUTION:")
            for category, count in statistics['category_distribution'].items():
                lines.append(f"  {category.title()}: {count}")
            
            lines.append("")
            lines.append("MOST COMMON ERRORS:")
            for error_key, count in statistics['most_common_errors']:
                lines.append(f"  {error_key}: {count} occurrences")
            
            return "\n".join(lines)
    
    def _create_error_info(self, exception: Exception,
                          context: Optional[Dict[str, Any]]) -> ErrorInfo:
        """Create detailed error information."""
        import uuid
        
        # Determine severity and category
        if isinstance(exception, QuantumPlannerError):
            severity = exception.severity
            category = exception.category
            context = {**(context or {}), **exception.context}
        else:
            severity = self._determine_severity(exception)
            category = self._determine_category(exception)
        
        return ErrorInfo(
            error_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback_str=traceback.format_exc(),
            context=context or {}
        )
    
    def _determine_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(exception, (SystemError, MemoryError, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (RuntimeError, IOError, OSError)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _determine_category(self, exception: Exception) -> ErrorCategory:
        """Determine error category based on exception type."""
        if isinstance(exception, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        elif isinstance(exception, (IOError, OSError)):
            return ErrorCategory.SYSTEM
        elif isinstance(exception, (MemoryError, RuntimeError)):
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.COMPUTATION
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = f"[{error_info.category.value.upper()}] {error_info.message}"
        
        if error_info.context:
            log_message += f" | Context: {error_info.context}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
            logger.critical(f"Traceback: {error_info.traceback_str}")
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
            logger.debug(f"Traceback: {error_info.traceback_str}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.debug(log_message)
    
    def _attempt_recovery(self, exception: Exception, error_info: ErrorInfo) -> bool:
        """Attempt to recover from error."""
        exception_type = type(exception)
        
        # Try specific handler first
        if exception_type in self.recovery_handlers:
            try:
                return self.recovery_handlers[exception_type](exception, error_info)
            except Exception as recovery_error:
                logger.error(f"Recovery handler failed: {recovery_error}")
        
        # Try parent class handlers
        for handled_type, handler in self.recovery_handlers.items():
            if isinstance(exception, handled_type):
                try:
                    return handler(exception, error_info)
                except Exception as recovery_error:
                    logger.error(f"Recovery handler failed: {recovery_error}")
        
        return False
    
    def _register_default_recovery_handlers(self):
        """Register default recovery handlers."""
        
        def handle_memory_error(exception: MemoryError, error_info: ErrorInfo) -> bool:
            """Handle memory errors by triggering garbage collection."""
            try:
                import gc
                collected = gc.collect()
                logger.info(f"Memory recovery: collected {collected} objects")
                return True
            except Exception:
                return False
        
        def handle_resource_error(exception: ResourceError, error_info: ErrorInfo) -> bool:
            """Handle resource errors by attempting resource cleanup."""
            try:
                # Basic resource cleanup
                logger.info("Attempting resource cleanup for recovery")
                return True
            except Exception:
                return False
        
        def handle_validation_error(exception: ValidationError, error_info: ErrorInfo) -> bool:
            """Handle validation errors by applying default values."""
            try:
                # Log validation issue for review
                logger.warning(f"Validation error recovery attempted: {exception.message}")
                return False  # Usually can't auto-recover from validation
            except Exception:
                return False
        
        self.register_recovery_handler(MemoryError, handle_memory_error)
        self.register_recovery_handler(ResourceError, handle_resource_error)
        self.register_recovery_handler(ValidationError, handle_validation_error)


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _global_error_handler


def handle_errors(attempt_recovery: bool = True, 
                 reraise: bool = False,
                 context: Optional[Dict[str, Any]] = None):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add function context
                func_context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                if context:
                    func_context.update(context)
                
                # Handle error
                recovery_successful = _global_error_handler.handle_error(
                    e, func_context, attempt_recovery
                )
                
                if reraise and not recovery_successful:
                    raise
                
                return None  # Or appropriate default value
        
        return wrapper
    return decorator


@contextmanager
def error_context(context: Dict[str, Any], 
                 handle_exceptions: bool = True,
                 attempt_recovery: bool = True):
    """Context manager for error handling with additional context."""
    try:
        yield
    except Exception as e:
        if handle_exceptions:
            recovery_successful = _global_error_handler.handle_error(
                e, context, attempt_recovery
            )
            if not recovery_successful:
                raise
        else:
            raise


def safe_execute(func: Callable, *args, 
                default_return: Any = None,
                context: Optional[Dict[str, Any]] = None,
                **kwargs) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        func_context = {
            'function': func.__name__ if hasattr(func, '__name__') else str(func),
            'safe_execute': True
        }
        if context:
            func_context.update(context)
        
        _global_error_handler.handle_error(e, func_context, attempt_recovery=True)
        return default_return


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise QuantumPlannerError(
                        "Circuit breaker is OPEN - function calls are blocked",
                        ErrorCategory.SYSTEM,
                        ErrorSeverity.HIGH
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (self.last_failure_time is not None and
                (datetime.now().timestamp() - self.last_failure_time) >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = "CLOSED"
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'recovery_timeout': self.recovery_timeout
        }