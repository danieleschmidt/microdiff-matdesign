"""Comprehensive logging configuration for MicroDiff-MatDesign."""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    SENSITIVE_KEYS = {
        'password', 'secret', 'key', 'token', 'auth', 'credential',
        'private', 'confidential', 'internal'
    }
    
    def filter(self, record):
        """Filter sensitive information from log records."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Simple check for sensitive information
            msg_lower = record.msg.lower()
            for sensitive_key in self.SENSITIVE_KEYS:
                if sensitive_key in msg_lower:
                    record.msg = f"[REDACTED - contains {sensitive_key}]"
                    break
        
        return True


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self._start_times[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and log the duration."""
        import time
        if operation not in self._start_times:
            self.logger.warning(f"Timer for '{operation}' was not started")
            return 0.0
        
        duration = time.time() - self._start_times[operation]
        del self._start_times[operation]
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.3f}s")
        return duration
    
    def log_memory_usage(self, operation: str = "current") -> None:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(
                f"Memory usage for {operation}: "
                f"RSS={memory_info.rss / 1024 / 1024:.1f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.1f}MB"
            )
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_security_filter: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default location)
        enable_console: Whether to enable console logging
        enable_security_filter: Whether to enable security filtering
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger('microdiff_matdesign')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file logging
    if log_file is None:
        # Create logs directory
        log_dir = Path.home() / '.microdiff_matdesign' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"microdiff_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    
    if enable_security_filter:
        file_handler.addFilter(SecurityFilter())
    
    logger.addHandler(file_handler)
    
    # Setup console logging
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        if enable_security_filter:
            console_handler.addFilter(SecurityFilter())
        
        logger.addHandler(console_handler)
    
    # Log startup information
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Security filtering: {'enabled' if enable_security_filter else 'disabled'}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (if None, returns root logger)
        
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger('microdiff_matdesign')
    else:
        return logging.getLogger(f'microdiff_matdesign.{name}')


def log_exception(logger: logging.Logger, operation: str, exception: Exception) -> None:
    """Log an exception with context.
    
    Args:
        logger: Logger instance
        operation: Description of the operation that failed
        exception: The exception that occurred
    """
    logger.error(
        f"Exception in {operation}: {type(exception).__name__}: {str(exception)}",
        exc_info=True
    )


def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]) -> None:
    """Log performance metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of performance metrics
    """
    metrics_str = ", ".join([f"{key}={value}" for key, value in metrics.items()])
    logger.info(f"Performance metrics: {metrics_str}")


class LoggingContextManager:
    """Context manager for operation logging."""
    
    def __init__(self, logger: logging.Logger, operation: str, log_args: bool = False):
        self.logger = logger
        self.operation = operation
        self.log_args = log_args
        self.perf_logger = PerformanceLogger(logger)
    
    def __enter__(self):
        self.logger.info(f"Starting operation: {self.operation}")
        self.perf_logger.start_timer(self.operation)
        self.perf_logger.log_memory_usage(f"{self.operation}_start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log_exception(self.logger, self.operation, exc_val)
            self.logger.error(f"Operation failed: {self.operation}")
        else:
            self.logger.info(f"Operation completed: {self.operation}")
        
        duration = self.perf_logger.end_timer(self.operation)
        self.perf_logger.log_memory_usage(f"{self.operation}_end")
        
        # Log performance summary
        log_performance_metrics(
            self.logger, 
            {
                'operation': self.operation,
                'duration_seconds': f"{duration:.3f}",
                'success': exc_type is None
            }
        )


def with_logging(operation_name: str = None, log_args: bool = False):
    """Decorator for automatic operation logging.
    
    Args:
        operation_name: Name of the operation (if None, uses function name)
        log_args: Whether to log function arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            op_name = operation_name or f"{func.__name__}"
            
            with LoggingContextManager(logger, op_name, log_args):
                if log_args and (args or kwargs):
                    args_str = ", ".join([str(arg) for arg in args])
                    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                    all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                    logger.debug(f"Function arguments: {all_args}")
                
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Initialize default logger
default_logger = setup_logging()


# Convenience functions
def debug(msg: str, *args, **kwargs):
    """Log debug message."""
    default_logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """Log info message."""
    default_logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """Log warning message."""
    default_logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """Log error message."""
    default_logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """Log critical message."""
    default_logger.critical(msg, *args, **kwargs)