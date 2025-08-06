"""Advanced logging configuration for quantum task planner."""

import logging
import logging.handlers
import json
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
import traceback
import os


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any]


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""
    
    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base event data
        event = LogEvent(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            extra_data={}
        )
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'message', 'exc_info', 'exc_text',
                          'stack_info', 'getMessage']:
                event.extra_data[key] = value
        
        # Add exception information
        if record.exc_info and self.include_traceback:
            event.extra_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(asdict(event), default=str, separators=(',', ':'))


class QuantumPlannerFormatter(logging.Formatter):
    """Custom formatter for quantum planner with color support."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True, include_context: bool = True):
        """Initialize formatter."""
        self.use_colors = use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
        self.include_context = include_context
        
        # Define format
        base_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        if self.include_context:
            base_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
        
        super().__init__(base_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional colors."""
        # Apply color if enabled
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        # Add extra context if available
        if hasattr(record, 'quantum_context'):
            formatted += f" | Quantum: {record.quantum_context}"
        
        if hasattr(record, 'task_id'):
            formatted += f" | Task: {record.task_id}"
        
        if hasattr(record, 'resource_id'):
            formatted += f" | Resource: {record.resource_id}"
        
        return formatted


class PerformanceLogFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_times = {}
        self._lock = threading.RLock()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance information to log record."""
        with self._lock:
            # Add thread and process info
            record.thread_id = threading.get_ident()
            record.process_id = os.getpid()
            
            # Add memory usage if available
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                record.memory_rss_mb = memory_info.rss / (1024 * 1024)
                record.memory_vms_mb = memory_info.vms / (1024 * 1024)
            except ImportError:
                pass
            except Exception:
                # Don't fail logging if memory info unavailable
                pass
        
        return True


class CircularBufferHandler(logging.Handler):
    """Log handler that maintains a circular buffer of recent log entries."""
    
    def __init__(self, capacity: int = 1000):
        """Initialize circular buffer handler."""
        super().__init__()
        self.capacity = capacity
        self.buffer: List[logging.LogRecord] = []
        self.index = 0
        self._lock = threading.RLock()
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to circular buffer."""
        with self._lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(record)
            else:
                self.buffer[self.index] = record
                self.index = (self.index + 1) % self.capacity
    
    def get_recent_logs(self, count: Optional[int] = None) -> List[logging.LogRecord]:
        """Get recent log entries."""
        with self._lock:
            if not self.buffer:
                return []
            
            if len(self.buffer) < self.capacity:
                # Buffer not full yet
                recent = self.buffer[:]
            else:
                # Buffer is full, get in chronological order
                recent = self.buffer[self.index:] + self.buffer[:self.index]
            
            if count:
                recent = recent[-count:]
            
            return recent
    
    def get_logs_by_level(self, level: int) -> List[logging.LogRecord]:
        """Get logs by minimum level."""
        with self._lock:
            return [record for record in self.get_recent_logs() 
                   if record.levelno >= level]
    
    def clear_buffer(self):
        """Clear the log buffer."""
        with self._lock:
            self.buffer.clear()
            self.index = 0


class LoggingConfig:
    """Centralized logging configuration for quantum planner."""
    
    def __init__(self, log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 enable_file_logging: bool = True,
                 enable_structured_logging: bool = False,
                 enable_console_colors: bool = True,
                 max_log_file_size: int = 10 * 1024 * 1024,  # 10MB
                 log_backup_count: int = 5):
        """Initialize logging configuration."""
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.enable_file_logging = enable_file_logging
        self.enable_structured_logging = enable_structured_logging
        self.enable_console_colors = enable_console_colors
        self.max_log_file_size = max_log_file_size
        self.log_backup_count = log_backup_count
        
        # Create circular buffer handler for recent logs
        self.circular_handler = CircularBufferHandler(capacity=1000)
        
        # Performance filter
        self.performance_filter = PerformanceLogFilter()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        if self.enable_structured_logging:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = QuantumPlannerFormatter(
                use_colors=self.enable_console_colors,
                include_context=True
            )
        
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(self.performance_filter)
        root_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file_logging:
            # Main log file with rotation
            main_log_file = self.log_dir / "quantum_planner.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=self.max_log_file_size,
                backupCount=self.log_backup_count
            )
            file_handler.setLevel(self.log_level)
            
            if self.enable_structured_logging:
                file_formatter = StructuredFormatter()
            else:
                file_formatter = QuantumPlannerFormatter(
                    use_colors=False,
                    include_context=True
                )
            
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(self.performance_filter)
            root_logger.addHandler(file_handler)
            
            # Error-specific log file
            error_log_file = self.log_dir / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_log_file_size,
                backupCount=self.log_backup_count
            )
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(file_formatter)
            error_handler.addFilter(self.performance_filter)
            root_logger.addHandler(error_handler)
            
            # Quantum-specific log file
            quantum_log_file = self.log_dir / "quantum_operations.log"
            quantum_handler = logging.handlers.RotatingFileHandler(
                quantum_log_file,
                maxBytes=self.max_log_file_size,
                backupCount=self.log_backup_count
            )
            quantum_handler.setLevel(logging.DEBUG)
            quantum_handler.addFilter(self._create_quantum_filter())
            quantum_handler.setFormatter(file_formatter)
            root_logger.addHandler(quantum_handler)
        
        # Add circular buffer handler
        self.circular_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(self.circular_handler)
        
        # Configure specific loggers
        self._configure_specific_loggers()
        
        logging.info("Logging configuration initialized")
    
    def _create_quantum_filter(self) -> logging.Filter:
        """Create filter for quantum-related logs."""
        class QuantumLogFilter(logging.Filter):
            def filter(self, record):
                return (
                    'quantum' in record.name.lower() or
                    'quantum' in record.getMessage().lower() or
                    hasattr(record, 'quantum_context')
                )
        
        return QuantumLogFilter()
    
    def _configure_specific_loggers(self):
        """Configure specific loggers for different components."""
        # Quantum engine logger
        quantum_logger = logging.getLogger('quantum_planner.core.quantum_engine')
        quantum_logger.setLevel(logging.DEBUG)
        
        # Scheduler logger
        scheduler_logger = logging.getLogger('quantum_planner.core.scheduler')
        scheduler_logger.setLevel(logging.INFO)
        
        # Algorithm loggers
        annealing_logger = logging.getLogger('quantum_planner.algorithms.quantum_annealing')
        annealing_logger.setLevel(logging.INFO)
        
        superposition_logger = logging.getLogger('quantum_planner.algorithms.superposition')
        superposition_logger.setLevel(logging.INFO)
        
        # Performance logger
        perf_logger = logging.getLogger('quantum_planner.utils.performance')
        perf_logger.setLevel(logging.INFO)
        
        # Validation logger
        validation_logger = logging.getLogger('quantum_planner.utils.validation')
        validation_logger.setLevel(logging.WARNING)
        
        # External library loggers (reduce noise)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('numpy').setLevel(logging.WARNING)
    
    def get_recent_logs(self, count: int = 100, 
                       level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent log entries as dictionaries."""
        if level:
            level_num = getattr(logging, level.upper())
            records = self.circular_handler.get_logs_by_level(level_num)
        else:
            records = self.circular_handler.get_recent_logs(count)
        
        logs = []
        for record in records[-count:]:
            log_dict = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add extra fields
            if hasattr(record, 'task_id'):
                log_dict['task_id'] = record.task_id
            if hasattr(record, 'resource_id'):
                log_dict['resource_id'] = record.resource_id
            if hasattr(record, 'quantum_context'):
                log_dict['quantum_context'] = record.quantum_context
            if hasattr(record, 'memory_rss_mb'):
                log_dict['memory_mb'] = record.memory_rss_mb
            
            logs.append(log_dict)
        
        return logs
    
    def export_logs(self, output_file: str, format_type: str = "json",
                   level: Optional[str] = None, count: int = 1000):
        """Export recent logs to file."""
        logs = self.get_recent_logs(count, level)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(logs, f, indent=2)
        elif format_type == "csv":
            import csv
            if logs:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                    writer.writeheader()
                    writer.writerows(logs)
        else:
            # Text format
            with open(output_path, 'w') as f:
                for log in logs:
                    f.write(f"{log['timestamp']} | {log['level']} | {log['logger']} | {log['message']}\n")
    
    def set_log_level(self, level: str, logger_name: Optional[str] = None):
        """Dynamically change log level."""
        level_num = getattr(logging, level.upper())
        
        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level_num)
        else:
            logging.getLogger().setLevel(level_num)
            # Update all handlers
            for handler in logging.getLogger().handlers:
                handler.setLevel(level_num)
        
        logging.info(f"Log level changed to {level} for {logger_name or 'root logger'}")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        recent_logs = self.circular_handler.get_recent_logs()
        
        if not recent_logs:
            return {'total_logs': 0}
        
        # Count by level
        level_counts = {}
        logger_counts = {}
        
        for record in recent_logs:
            level_counts[record.levelname] = level_counts.get(record.levelname, 0) + 1
            logger_counts[record.name] = logger_counts.get(record.name, 0) + 1
        
        # Recent activity (last hour)
        now = datetime.now().timestamp()
        recent_count = len([r for r in recent_logs 
                           if (now - r.created) <= 3600])
        
        return {
            'total_logs': len(recent_logs),
            'recent_logs_1h': recent_count,
            'level_distribution': level_counts,
            'logger_distribution': logger_counts,
            'circular_buffer_capacity': self.circular_handler.capacity,
            'log_files': list(self.log_dir.glob("*.log")) if self.log_dir.exists() else []
        }


# Context managers and utilities
class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, **context):
        """Initialize with context data."""
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        """Enter context."""
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        logging.setLogRecordFactory(self.old_factory)


def log_quantum_operation(operation_name: str):
    """Decorator to log quantum operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            
            with LogContext(quantum_context=operation_name):
                logger.debug(f"Starting quantum operation: {operation_name}")
                try:
                    result = func(*args, **kwargs)
                    logger.debug(f"Completed quantum operation: {operation_name}")
                    return result
                except Exception as e:
                    logger.error(f"Failed quantum operation: {operation_name} - {e}")
                    raise
        
        return wrapper
    return decorator


def log_task_operation(func):
    """Decorator to log task operations with task context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Try to extract task_id from arguments
        task_id = None
        if args and hasattr(args[0], 'id'):
            task_id = args[0].id
        elif 'task_id' in kwargs:
            task_id = kwargs['task_id']
        elif 'task' in kwargs and hasattr(kwargs['task'], 'id'):
            task_id = kwargs['task'].id
        
        context = {}
        if task_id:
            context['task_id'] = task_id
        
        with LogContext(**context):
            logger.debug(f"Task operation: {func.__name__}")
            return func(*args, **kwargs)
    
    return wrapper


# Global logging configuration instance
_logging_config = None


def setup_logging(log_level: str = "INFO", 
                 log_dir: Optional[str] = None,
                 **kwargs) -> LoggingConfig:
    """Setup global logging configuration."""
    global _logging_config
    _logging_config = LoggingConfig(log_level, log_dir, **kwargs)
    return _logging_config


def get_logging_config() -> Optional[LoggingConfig]:
    """Get global logging configuration."""
    return _logging_config