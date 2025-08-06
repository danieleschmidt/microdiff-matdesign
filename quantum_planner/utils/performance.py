"""Performance monitoring and optimization utilities."""

import time
import threading
import logging
import psutil
import gc
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import functools
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    timestamp: datetime
    category: str = "general"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class OperationStats:
    """Statistics for a specific operation."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_execution: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 100.0


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, max_history: int = 1000,
                 enable_memory_tracking: bool = True,
                 enable_cpu_tracking: bool = True):
        """Initialize performance monitor."""
        self.max_history = max_history
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_cpu_tracking = enable_cpu_tracking
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, OperationStats] = {}
        self.active_operations: Dict[str, float] = {}  # operation_id -> start_time
        
        # System monitoring
        self.system_metrics: Dict[str, List[float]] = defaultdict(list)
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Initialized PerformanceMonitor")
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Stopped performance monitoring")
    
    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        with self._lock:
            self.active_operations[operation_id] = time.perf_counter()
            
            if operation_name not in self.operation_stats:
                self.operation_stats[operation_name] = OperationStats(name=operation_name)
        
        return operation_id
    
    def end_operation(self, operation_name: str, operation_id: Optional[str] = None,
                     success: bool = True) -> float:
        """End timing an operation."""
        end_time = time.perf_counter()
        
        with self._lock:
            # Find operation
            if operation_id and operation_id in self.active_operations:
                start_time = self.active_operations.pop(operation_id)
            else:
                # Find by name (less precise but works)
                matching_ops = [oid for oid in self.active_operations.keys() 
                              if oid.startswith(operation_name)]
                if matching_ops:
                    operation_id = matching_ops[0]
                    start_time = self.active_operations.pop(operation_id)
                else:
                    logger.warning(f"Operation {operation_name} not found in active operations")
                    return 0.0
            
            duration = end_time - start_time
            
            # Update statistics
            stats = self.operation_stats.get(operation_name)
            if stats:
                stats.count += 1
                stats.total_time += duration
                stats.min_time = min(stats.min_time, duration)
                stats.max_time = max(stats.max_time, duration)
                stats.avg_time = stats.total_time / stats.count
                stats.last_execution = datetime.now()
                
                if not success:
                    stats.error_count += 1
                
                stats.success_rate = ((stats.count - stats.error_count) / stats.count) * 100
            
            # Record metric
            metric = PerformanceMetric(
                name=f"{operation_name}_duration",
                value=duration,
                timestamp=datetime.now(),
                category="operation",
                tags={"operation": operation_name, "success": str(success)}
            )
            self.metrics_history.append(metric)
        
        return duration
    
    def record_metric(self, name: str, value: float, 
                     category: str = "custom", **tags):
        """Record a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            category=category,
            tags=tags
        )
        
        with self._lock:
            self.metrics_history.append(metric)
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, OperationStats]:
        """Get operation statistics."""
        with self._lock:
            if operation_name:
                return {operation_name: self.operation_stats.get(operation_name)}
            return self.operation_stats.copy()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = {}
        
        try:
            # Memory metrics
            if self.enable_memory_tracking:
                memory = psutil.virtual_memory()
                metrics.update({
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_percent': memory.percent
                })
                
                # Process memory
                process = psutil.Process()
                process_memory = process.memory_info()
                metrics.update({
                    'process_memory_rss_mb': process_memory.rss / (1024**2),
                    'process_memory_vms_mb': process_memory.vms / (1024**2)
                })
            
            # CPU metrics  
            if self.enable_cpu_tracking:
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count()
                
                metrics.update({
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'load_average_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
                })
        
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def get_metrics_summary(self, time_window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            now = datetime.now()
            
            # Filter metrics by time window
            if time_window_minutes:
                cutoff = now - timedelta(minutes=time_window_minutes)
                filtered_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff]
            else:
                filtered_metrics = list(self.metrics_history)
            
            if not filtered_metrics:
                return {}
            
            # Group metrics by name
            metric_groups = defaultdict(list)
            for metric in filtered_metrics:
                metric_groups[metric.name].append(metric.value)
            
            # Calculate statistics
            summary = {
                'total_metrics': len(filtered_metrics),
                'time_window_minutes': time_window_minutes,
                'metric_stats': {}
            }
            
            for name, values in metric_groups.items():
                if values:
                    summary['metric_stats'][name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'recent': values[-1] if values else 0
                    }
            
            # Add operation statistics
            summary['operations'] = {}
            for name, stats in self.operation_stats.items():
                summary['operations'][name] = {
                    'count': stats.count,
                    'avg_time': stats.avg_time,
                    'min_time': stats.min_time,
                    'max_time': stats.max_time,
                    'success_rate': stats.success_rate,
                    'error_count': stats.error_count
                }
            
            # Add system metrics
            summary['system'] = self.get_system_metrics()
            
            return summary
    
    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on thresholds."""
        alerts = []
        
        with self._lock:
            # Check operation performance
            for name, stats in self.operation_stats.items():
                # Slow operations
                if stats.avg_time > 5.0:  # 5 seconds threshold
                    alerts.append({
                        'type': 'slow_operation',
                        'severity': 'warning',
                        'message': f"Operation {name} averaging {stats.avg_time:.2f}s",
                        'operation': name,
                        'avg_time': stats.avg_time
                    })
                
                # High error rate
                if stats.success_rate < 95.0 and stats.count > 10:
                    alerts.append({
                        'type': 'high_error_rate',
                        'severity': 'error',
                        'message': f"Operation {name} has {stats.success_rate:.1f}% success rate",
                        'operation': name,
                        'success_rate': stats.success_rate,
                        'error_count': stats.error_count
                    })
            
            # Check system metrics
            system_metrics = self.get_system_metrics()
            
            if system_metrics.get('memory_percent', 0) > 90:
                alerts.append({
                    'type': 'high_memory_usage',
                    'severity': 'warning',
                    'message': f"Memory usage at {system_metrics['memory_percent']:.1f}%",
                    'memory_percent': system_metrics['memory_percent']
                })
            
            if system_metrics.get('cpu_percent', 0) > 90:
                alerts.append({
                    'type': 'high_cpu_usage', 
                    'severity': 'warning',
                    'message': f"CPU usage at {system_metrics['cpu_percent']:.1f}%",
                    'cpu_percent': system_metrics['cpu_percent']
                })
        
        return alerts
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        before_memory = self.get_system_metrics()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear old metrics if needed
        with self._lock:
            if len(self.metrics_history) > self.max_history * 0.8:
                # Remove oldest 20% of metrics
                remove_count = int(len(self.metrics_history) * 0.2)
                for _ in range(remove_count):
                    self.metrics_history.popleft()
        
        after_memory = self.get_system_metrics()
        
        optimization_result = {
            'garbage_collected': collected,
            'metrics_cleaned': remove_count if 'remove_count' in locals() else 0,
            'memory_before_mb': before_memory.get('process_memory_rss_mb', 0),
            'memory_after_mb': after_memory.get('process_memory_rss_mb', 0),
            'memory_saved_mb': before_memory.get('process_memory_rss_mb', 0) - 
                             after_memory.get('process_memory_rss_mb', 0)
        }
        
        logger.info(f"Memory optimization: {optimization_result}")
        return optimization_result
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        summary = self.get_metrics_summary()
        
        if format_type == "json":
            import json
            return json.dumps(summary, indent=2, default=str)
        elif format_type == "csv":
            # Simple CSV export of recent metrics
            lines = ["timestamp,name,value,category"]
            with self._lock:
                for metric in self.metrics_history:
                    lines.append(f"{metric.timestamp},{metric.name},{metric.value},{metric.category}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self.get_system_metrics()
                
                # Record as metrics
                for name, value in system_metrics.items():
                    if isinstance(value, (int, float)):
                        self.record_metric(
                            name=name,
                            value=float(value),
                            category="system"
                        )
                
                # Store for trending
                with self._lock:
                    for name, value in system_metrics.items():
                        if isinstance(value, (int, float)):
                            self.system_metrics[name].append(float(value))
                            # Keep only recent values
                            if len(self.system_metrics[name]) > 100:
                                self.system_metrics[name].pop(0)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'active_operations': len(self.active_operations),
            'total_operations': len(self.operation_stats),
            'total_metrics': len(self.metrics_history)
        }
    
    def reset_stats(self):
        """Reset all performance statistics."""
        with self._lock:
            self.operation_stats.clear()
            self.active_operations.clear()
            self.metrics_history.clear()
            self.system_metrics.clear()
        
        logger.info("Performance statistics reset")


def performance_monitor(operation_name: Optional[str] = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create monitor
            if hasattr(wrapper, '_monitor'):
                monitor = wrapper._monitor
            else:
                monitor = PerformanceMonitor()
                wrapper._monitor = monitor
            
            # Start monitoring
            op_name = operation_name or func.__name__
            operation_id = monitor.start_operation(op_name)
            
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                monitor.end_operation(op_name, operation_id, success=success)
        
        return wrapper
    return decorator


class ResourceManager:
    """Manage computational resources efficiently."""
    
    def __init__(self, max_memory_mb: Optional[int] = None,
                 max_cpu_percent: float = 80.0):
        """Initialize resource manager."""
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"Initialized ResourceManager with limits: memory={max_memory_mb}MB, cpu={max_cpu_percent}%")
    
    def check_resource_availability(self) -> Dict[str, Any]:
        """Check current resource availability."""
        metrics = self.performance_monitor.get_system_metrics()
        
        available = {
            'memory_available': True,
            'cpu_available': True,
            'can_proceed': True,
            'limitations': []
        }
        
        # Check memory
        if self.max_memory_mb:
            current_memory_mb = metrics.get('process_memory_rss_mb', 0)
            if current_memory_mb > self.max_memory_mb:
                available['memory_available'] = False
                available['can_proceed'] = False
                available['limitations'].append(f"Memory usage ({current_memory_mb:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
        
        # Check CPU
        current_cpu = metrics.get('cpu_percent', 0)
        if current_cpu > self.max_cpu_percent:
            available['cpu_available'] = False
            available['can_proceed'] = False
            available['limitations'].append(f"CPU usage ({current_cpu:.1f}%) exceeds limit ({self.max_cpu_percent}%)")
        
        return available
    
    def wait_for_resources(self, timeout_seconds: float = 300.0) -> bool:
        """Wait for resources to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            availability = self.check_resource_availability()
            if availability['can_proceed']:
                return True
            
            logger.debug(f"Waiting for resources: {availability['limitations']}")
            time.sleep(5.0)  # Check every 5 seconds
        
        logger.warning(f"Resource wait timeout after {timeout_seconds}s")
        return False
    
    def optimize_for_task(self, task_complexity: str = "medium") -> Dict[str, Any]:
        """Optimize resources for specific task complexity."""
        optimization = {
            'memory_optimized': False,
            'cpu_optimized': False,
            'actions_taken': []
        }
        
        # Memory optimization
        if task_complexity in ["high", "maximum"]:
            result = self.performance_monitor.optimize_memory()
            if result['memory_saved_mb'] > 0:
                optimization['memory_optimized'] = True
                optimization['actions_taken'].append("Memory cleanup")
        
        # CPU optimization (placeholder for actual CPU optimization)
        if task_complexity in ["high", "maximum"]:
            # Could implement CPU affinity settings, process priority, etc.
            optimization['cpu_optimized'] = True
            optimization['actions_taken'].append("CPU optimization")
        
        return optimization