"""Performance optimization module for MicroDiff-MatDesign.

This module provides comprehensive performance optimization and scaling
capabilities for production deployment of diffusion models.

Features:
    - Performance profiling and monitoring
    - Memory optimization techniques
    - Intelligent caching systems
    - Parallel and distributed processing
    - Auto-scaling capabilities
"""

from .optimization import (
    PerformanceProfiler,
    MemoryOptimizer, 
    SmartCache,
    ParallelProcessingManager,
    AutoScaler,
    PerformanceOptimizedPipeline,
    memoize_expensive_computation,
    batch_processor
)

__all__ = [
    'PerformanceProfiler',
    'MemoryOptimizer',
    'SmartCache', 
    'ParallelProcessingManager',
    'AutoScaler',
    'PerformanceOptimizedPipeline',
    'memoize_expensive_computation',
    'batch_processor'
]