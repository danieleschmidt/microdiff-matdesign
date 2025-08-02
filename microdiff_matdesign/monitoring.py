"""
Monitoring and observability utilities for MicroDiff-MatDesign.

This module provides health checks, metrics collection, and logging utilities
for monitoring the application in production environments.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        extra_fields = ['request_id', 'user_id', 'duration_ms', 'model_type', 'alloy']
        for field in extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
                
        return json.dumps(log_entry)


class HealthChecker:
    """Application health check utilities."""
    
    @staticmethod
    def check_gpu_availability() -> bool:
        """Check if GPU is available for model inference."""
        if not TORCH_AVAILABLE:
            return False
        try:
            return torch.cuda.is_available()
        except Exception:
            return False
    
    @staticmethod
    def check_model_loading() -> bool:
        """Check if pre-trained models can be loaded."""
        try:
            # Attempt to import and instantiate a basic model
            from .core import MicrostructureDiffusion
            model = MicrostructureDiffusion(alloy="Ti-6Al-4V", pretrained=False)
            return True
        except Exception:
            return False
    
    @staticmethod
    def check_storage_access() -> bool:
        """Check if storage is accessible."""
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(b"health check")
                tmp.flush()
                
            # Try to read the file back
            with open(tmp.name, 'rb') as f:
                content = f.read()
                
            os.unlink(tmp.name)
            return content == b"health check"
        except Exception:
            return False
    
    @classmethod
    def get_health_status(cls) -> Dict[str, Any]:
        """Get comprehensive health status."""
        checks = {
            'gpu': cls.check_gpu_availability(),
            'model': cls.check_model_loading(),
            'storage': cls.check_storage_access(),
        }
        
        overall_status = 'healthy' if all(checks.values()) else 'degraded'
        
        return {
            'status': overall_status,
            'checks': checks,
            'timestamp': datetime.utcnow().isoformat()
        }


class MetricsCollector:
    """Prometheus metrics collector."""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            self._metrics_enabled = False
            return
            
        self._metrics_enabled = True
        
        # Define metrics
        self.request_count = Counter(
            'microdiff_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'microdiff_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.model_inference_duration = Histogram(
            'microdiff_model_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_type', 'alloy']
        )
        
        self.active_connections = Gauge(
            'microdiff_active_connections',
            'Number of active connections'
        )
        
        self.gpu_memory_usage = Gauge(
            'microdiff_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['device']
        )
        
        self.model_accuracy = Gauge(
            'microdiff_model_accuracy',
            'Model prediction accuracy',
            ['model_type', 'validation_set']
        )
    
    def track_request(self, method: str, endpoint: str, status: str) -> None:
        """Track request metrics."""
        if self._metrics_enabled:
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
    
    def track_request_duration(self, method: str, endpoint: str, duration: float) -> None:
        """Track request duration metrics."""
        if self._metrics_enabled:
            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
    
    def track_model_inference(self, model_type: str, alloy: str, duration: float) -> None:
        """Track model inference duration."""
        if self._metrics_enabled:
            self.model_inference_duration.labels(
                model_type=model_type,
                alloy=alloy
            ).observe(duration)
    
    def update_gpu_metrics(self) -> None:
        """Update GPU usage metrics."""
        if not self._metrics_enabled or not TORCH_AVAILABLE:
            return
            
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i)
                    self.gpu_memory_usage.labels(device=f'cuda:{i}').set(memory_used)
        except Exception:
            pass
    
    def update_model_accuracy(self, model_type: str, validation_set: str, accuracy: float) -> None:
        """Update model accuracy metrics."""
        if self._metrics_enabled:
            self.model_accuracy.labels(
                model_type=model_type,
                validation_set=validation_set
            ).set(accuracy)
    
    def start_metrics_server(self, port: int = 8000) -> None:
        """Start Prometheus metrics server."""
        if self._metrics_enabled:
            start_http_server(port)


# Global metrics collector instance
metrics = MetricsCollector()


def track_request_metrics(method: str = 'unknown', endpoint: str = 'unknown'):
    """Decorator to track request metrics."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                metrics.track_request_duration(method, endpoint, duration)
                metrics.track_request(method, endpoint, status)
        
        return wrapper
    return decorator


def track_model_inference(model_type: str, alloy: str):
    """Decorator to track model inference metrics."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.track_model_inference(model_type, alloy, duration)
        
        return wrapper
    return decorator


def setup_logging(log_level: str = "INFO", use_json: bool = True) -> None:
    """Setup structured logging configuration."""
    
    # Create formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    loggers = [
        'microdiff_matdesign',
        'microdiff_matdesign.models',
        'microdiff_matdesign.imaging',
        'microdiff_matdesign.optimization'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))


def log_model_inference(model_type: str, duration_ms: float, success: bool, 
                       request_id: Optional[str] = None):
    """Log model inference metrics with structured data."""
    logger = logging.getLogger('microdiff_matdesign.models')
    
    extra = {
        'model_type': model_type,
        'duration_ms': duration_ms,
        'success': success
    }
    
    if request_id:
        extra['request_id'] = request_id
    
    if success:
        logger.info("Model inference completed successfully", extra=extra)
    else:
        logger.error("Model inference failed", extra=extra)


class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger('microdiff_matdesign.performance')
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.info(
                f"Performance: {self.name}",
                extra={'duration_ms': duration * 1000, 'operation': self.name}
            )


def profile_operation(name: str):
    """Decorator for profiling operation performance."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            with PerformanceProfiler(name):
                return f(*args, **kwargs)
        return wrapper
    return decorator