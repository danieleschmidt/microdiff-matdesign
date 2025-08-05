"""Monitoring and health check system for MicroDiff-MatDesign."""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .logging_config import get_logger
from .error_handling import MicroDiffError, ErrorSeverity


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    description: str
    critical: bool = False
    interval: float = 60.0  # seconds
    timeout: float = 30.0   # seconds
    last_run: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    gpu_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Application performance metrics."""
    timestamp: datetime
    operations_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    active_operations: int = 0
    queue_size: int = 0
    cache_hit_rate: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class SystemMonitor:
    """System monitoring and health checking."""
    
    def __init__(self, monitoring_interval: float = 30.0):
        self.logger = get_logger('system_monitor')
        self.monitoring_interval = monitoring_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        self.alerts_enabled = True
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'disk_percent': 90.0,
            'error_rate': 5.0  # 5% error rate
        }
    
    def add_health_check(self, check: HealthCheck) -> None:
        """Add a health check.
        
        Args:
            check: Health check to add
        """
        self.health_checks[check.name] = check
        self.logger.info(f"Added health check: {check.name}")
    
    def remove_health_check(self, name: str) -> None:
        """Remove a health check.
        
        Args:
            name: Name of health check to remove
        """
        if name in self.health_checks:
            del self.health_checks[name]
            self.logger.info(f"Removed health check: {name}")
    
    def run_health_check(self, name: str) -> bool:
        """Run a specific health check.
        
        Args:
            name: Name of health check to run
            
        Returns:
            True if check passed, False otherwise
        """
        if name not in self.health_checks:
            self.logger.error(f"Health check '{name}' not found")
            return False
        
        check = self.health_checks[name]
        
        try:
            # Run with timeout
            import signal
            
            class TimeoutError(Exception):
                pass
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Health check timed out")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(check.timeout))
            
            try:
                result = check.check_function()
                check.last_result = result
                check.last_run = datetime.now()
                check.last_error = None
                
                if result:
                    self.logger.debug(f"Health check '{name}' passed")
                else:
                    self.logger.warning(f"Health check '{name}' failed")
                
                return result
                
            finally:
                signal.alarm(0)  # Disable alarm
                
        except Exception as e:
            check.last_result = False
            check.last_run = datetime.now()
            check.last_error = str(e)
            
            if check.critical:
                self.logger.critical(f"Critical health check '{name}' failed: {e}")
            else:
                self.logger.error(f"Health check '{name}' error: {e}")
            
            return False
    
    def run_all_health_checks(self) -> Dict[str, bool]:
        """Run all health checks.
        
        Returns:
            Dictionary mapping check names to results
        """
        results = {}
        
        for name, check in self.health_checks.items():
            # Check if it's time to run this check
            if (check.last_run is None or 
                datetime.now() - check.last_run > timedelta(seconds=check.interval)):
                results[name] = self.run_health_check(name)
            else:
                results[name] = check.last_result if check.last_result is not None else True
        
        return results
    
    def get_system_health(self) -> HealthStatus:
        """Get overall system health status.
        
        Returns:
            Overall health status
        """
        check_results = self.run_all_health_checks()
        
        # Check for critical failures
        for name, result in check_results.items():
            check = self.health_checks[name]
            if check.critical and not result:
                return HealthStatus.CRITICAL
        
        # Check for any failures
        if not all(check_results.values()):
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics.
        
        Returns:
            Current system metrics
        """
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage for current directory
            disk = psutil.disk_usage('.')
            
            # Network stats
            network = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_percent=(disk.used / disk.total) * 100,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv
            )
            
            # Try to get GPU metrics
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    
                    metrics.gpu_memory_total_mb = gpu_memory / (1024 * 1024)
                    metrics.gpu_memory_used_mb = gpu_memory_used / (1024 * 1024)
                    metrics.gpu_percent = (gpu_memory_used / gpu_memory) * 100
            except ImportError:
                pass
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_free_gb=0.0
            )
    
    def record_system_metrics(self) -> None:
        """Record current system metrics to history."""
        metrics = self.collect_system_metrics()
        self.metrics_history.append(metrics)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        # Check for alerts
        if self.alerts_enabled:
            self._check_metric_alerts(metrics)
    
    def record_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record application performance metrics.
        
        Args:
            metrics: Performance metrics to record
        """
        self.performance_history.append(metrics)
        
        # Trim history if too large
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
        # Check for performance alerts
        if self.alerts_enabled and metrics.error_rate > self.alert_thresholds['error_rate']:
            self.logger.warning(f"High error rate detected: {metrics.error_rate:.1f}%")
    
    def _check_metric_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against alert thresholds.
        
        Args:
            metrics: System metrics to check
        """
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            self.logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            self.logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            self.logger.warning(f"High disk usage: {metrics.disk_percent:.1f}%")
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring in background thread."""
        if self.running:
            self.logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started system monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped system monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self.record_system_metrics()
                self.run_all_health_checks()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report.
        
        Returns:
            Dictionary with monitoring information
        """
        # Latest metrics
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        latest_performance = self.performance_history[-1] if self.performance_history else None
        
        # Health check summary
        health_summary = {}
        for name, check in self.health_checks.items():
            health_summary[name] = {
                'status': check.last_result,
                'last_run': check.last_run.isoformat() if check.last_run else None,
                'error': check.last_error,
                'critical': check.critical
            }
        
        # Calculate averages from recent history
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': self.get_system_health().value,
            'monitoring_status': 'running' if self.running else 'stopped',
            'system_metrics': {
                'current': latest_metrics.__dict__ if latest_metrics else None,
                'averages': {
                    'cpu_percent': avg_cpu,
                    'memory_percent': avg_memory
                }
            },
            'performance_metrics': latest_performance.__dict__ if latest_performance else None,
            'health_checks': health_summary,
            'metrics_history_size': len(self.metrics_history),
            'performance_history_size': len(self.performance_history)
        }


class PerformanceTracker:
    """Track application performance metrics."""
    
    def __init__(self):
        self.logger = get_logger('performance_tracker')
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.start_time = time.time()
        self.active_operations = 0
        self.lock = threading.Lock()
    
    def record_operation(self, operation: str, duration: float, success: bool = True) -> None:
        """Record an operation's performance.
        
        Args:
            operation: Name of the operation
            duration: Operation duration in seconds
            success: Whether the operation succeeded
        """
        with self.lock:
            if operation not in self.operation_times:
                self.operation_times[operation] = []
                self.operation_counts[operation] = 0
                self.error_counts[operation] = 0
            
            self.operation_times[operation].append(duration)
            self.operation_counts[operation] += 1
            
            if not success:
                self.error_counts[operation] += 1
            
            # Keep only recent times (last 100)
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation].pop(0)
    
    def start_operation(self) -> None:
        """Mark start of an operation."""
        with self.lock:
            self.active_operations += 1
    
    def end_operation(self) -> None:
        """Mark end of an operation."""
        with self.lock:
            self.active_operations = max(0, self.active_operations - 1)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics.
        
        Returns:
            Current performance metrics
        """
        with self.lock:
            uptime = time.time() - self.start_time
            total_operations = sum(self.operation_counts.values())
            total_errors = sum(self.error_counts.values())
            
            # Calculate operations per second
            ops_per_second = total_operations / uptime if uptime > 0 else 0
            
            # Calculate average response time
            all_times = []
            for times in self.operation_times.values():
                all_times.extend(times)
            avg_response_time = sum(all_times) / len(all_times) if all_times else 0
            
            # Calculate error rate
            error_rate = (total_errors / total_operations * 100) if total_operations > 0 else 0
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                operations_per_second=ops_per_second,
                average_response_time=avg_response_time,
                error_rate=error_rate,
                active_operations=self.active_operations,
                queue_size=0,  # Would need to be updated by application
                cache_hit_rate=0.0  # Would need to be updated by application
            )
    
    def get_operation_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Operation statistics or None if not found
        """
        with self.lock:
            if operation not in self.operation_counts:
                return None
            
            times = self.operation_times[operation]
            count = self.operation_counts[operation]
            errors = self.error_counts[operation]
            
            return {
                'operation': operation,
                'total_count': count,
                'error_count': errors,
                'error_rate': (errors / count * 100) if count > 0 else 0,
                'average_time': sum(times) / len(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0,
                'recent_times': times[-10:]  # Last 10 times
            }


# Global instances
system_monitor = SystemMonitor()
performance_tracker = PerformanceTracker()


def setup_default_health_checks():
    """Setup default health checks."""
    
    def check_disk_space() -> bool:
        """Check if sufficient disk space is available."""
        disk = psutil.disk_usage('.')
        free_percent = (disk.free / disk.total) * 100
        return free_percent > 10.0  # At least 10% free
    
    def check_memory_usage() -> bool:
        """Check if memory usage is reasonable."""
        memory = psutil.virtual_memory()
        return memory.percent < 95.0  # Less than 95% used
    
    def check_process_health() -> bool:
        """Check if the current process is healthy."""
        try:
            process = psutil.Process()
            # Check if process is responsive (this is a simple check)
            process.cpu_percent()
            return True
        except Exception:
            return False
    
    # Add health checks
    system_monitor.add_health_check(HealthCheck(
        name="disk_space",
        check_function=check_disk_space,
        description="Check available disk space",
        critical=True,
        interval=300.0  # 5 minutes
    ))
    
    system_monitor.add_health_check(HealthCheck(
        name="memory_usage",
        check_function=check_memory_usage,
        description="Check memory usage",
        critical=True,
        interval=60.0  # 1 minute
    ))
    
    system_monitor.add_health_check(HealthCheck(
        name="process_health",
        check_function=check_process_health,
        description="Check process health",
        critical=False,
        interval=30.0  # 30 seconds
    ))


# Initialize default health checks
setup_default_health_checks()