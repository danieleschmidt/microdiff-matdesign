"""Performance testing configuration and utilities for MicroDiff-MatDesign."""

import time
import psutil
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""
    
    execution_time: float
    memory_used_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    
    def meets_requirements(self, requirements: Dict[str, float]) -> bool:
        """Check if metrics meet performance requirements."""
        checks = {
            "max_time": self.execution_time <= requirements.get("max_time", float('inf')),
            "max_memory": self.memory_used_mb <= requirements.get("max_memory_mb", float('inf')),
            "max_cpu": self.cpu_percent <= requirements.get("max_cpu_percent", 100.0)
        }
        
        if self.gpu_memory_mb is not None and "max_gpu_memory_mb" in requirements:
            checks["max_gpu_memory"] = self.gpu_memory_mb <= requirements["max_gpu_memory_mb"]
            
        return all(checks.values())


@contextmanager
def performance_monitor(sample_interval: float = 0.1):
    """Context manager to monitor performance during test execution."""
    process = psutil.Process(os.getpid())
    
    # Initial measurements
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    cpu_samples = []
    
    # GPU memory (if available)
    gpu_memory_start = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    except ImportError:
        pass
    
    try:
        # Start monitoring
        yield
        
    finally:
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        execution_time = end_time - start_time
        memory_used = max(0, end_memory - start_memory)
        
        # CPU usage (approximate)
        cpu_percent = process.cpu_percent()
        
        # GPU memory (if available)
        gpu_memory_used = None
        if gpu_memory_start is not None:
            try:
                torch.cuda.synchronize()
                gpu_memory_end = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_used = max(0, gpu_memory_end - gpu_memory_start)
            except:
                pass
        
        # Store metrics globally for test access
        global _last_performance_metrics
        _last_performance_metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_used_mb=memory_used,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory_used
        )


def get_last_performance_metrics() -> Optional[PerformanceMetrics]:
    """Get the last recorded performance metrics."""
    global _last_performance_metrics
    return getattr(get_last_performance_metrics, '_last_performance_metrics', None)


# Performance requirements for different test types
PERFORMANCE_REQUIREMENTS = {
    "unit_test": {
        "max_time": 1.0,  # 1 second
        "max_memory_mb": 100.0,  # 100 MB
        "max_cpu_percent": 80.0
    },
    "integration_test": {
        "max_time": 10.0,  # 10 seconds  
        "max_memory_mb": 500.0,  # 500 MB
        "max_cpu_percent": 90.0
    },
    "inference_test": {
        "max_time": 5.0,  # 5 seconds
        "max_memory_mb": 1000.0,  # 1 GB
        "max_gpu_memory_mb": 2000.0,  # 2 GB GPU
        "max_cpu_percent": 95.0
    },
    "training_test": {
        "max_time": 30.0,  # 30 seconds for small training
        "max_memory_mb": 2000.0,  # 2 GB
        "max_gpu_memory_mb": 4000.0,  # 4 GB GPU
        "max_cpu_percent": 95.0
    }
}


def assert_performance_requirements(test_type: str, custom_requirements: Optional[Dict[str, float]] = None):
    """Assert that the last recorded performance metrics meet requirements."""
    metrics = get_last_performance_metrics()
    if metrics is None:
        raise ValueError("No performance metrics recorded. Use performance_monitor() context.")
    
    requirements = PERFORMANCE_REQUIREMENTS.get(test_type, {})
    if custom_requirements:
        requirements.update(custom_requirements)
    
    if not metrics.meets_requirements(requirements):
        error_msg = f"Performance requirements not met for {test_type}:\n"
        error_msg += f"  Execution time: {metrics.execution_time:.3f}s (max: {requirements.get('max_time', 'unlimited')})\n"
        error_msg += f"  Memory used: {metrics.memory_used_mb:.1f}MB (max: {requirements.get('max_memory_mb', 'unlimited')})\n"
        error_msg += f"  CPU usage: {metrics.cpu_percent:.1f}% (max: {requirements.get('max_cpu_percent', 'unlimited')})\n"
        if metrics.gpu_memory_mb is not None:
            error_msg += f"  GPU memory: {metrics.gpu_memory_mb:.1f}MB (max: {requirements.get('max_gpu_memory_mb', 'unlimited')})\n"
        
        raise AssertionError(error_msg)


# Benchmark utilities
class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.times = []
        self.memory_usage = []
        
    def add_sample(self, time_taken: float, memory_mb: float):
        """Add a benchmark sample."""
        self.times.append(time_taken)
        self.memory_usage.append(memory_mb)
    
    @property
    def mean_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0
    
    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0.0
    
    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0.0
    
    @property
    def mean_memory(self) -> float:
        return sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0.0
    
    def summary(self) -> Dict[str, Any]:
        """Get benchmark summary statistics."""
        return {
            "name": self.name,
            "samples": len(self.times),
            "time_stats": {
                "mean": self.mean_time,
                "min": self.min_time,
                "max": self.max_time,
                "std": (sum((t - self.mean_time) ** 2 for t in self.times) / len(self.times)) ** 0.5 if len(self.times) > 1 else 0.0
            },
            "memory_stats": {
                "mean": self.mean_memory,
                "max": max(self.memory_usage) if self.memory_usage else 0.0
            }
        }


@contextmanager
def benchmark_test(name: str, num_samples: int = 5):
    """Context manager for running benchmark tests."""
    result = BenchmarkResult(name)
    
    for i in range(num_samples):
        with performance_monitor():
            yield result
        
        metrics = get_last_performance_metrics()
        if metrics:
            result.add_sample(metrics.execution_time, metrics.memory_used_mb)
    
    # Store result globally
    global _last_benchmark_result
    _last_benchmark_result = result


def get_last_benchmark_result() -> Optional[BenchmarkResult]:
    """Get the last benchmark result."""
    return getattr(get_last_benchmark_result, '_last_benchmark_result', None)


# Initialize globals
_last_performance_metrics = None
_last_benchmark_result = None
