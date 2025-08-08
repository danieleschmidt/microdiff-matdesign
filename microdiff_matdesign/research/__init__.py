"""Research module for advanced diffusion model techniques.

This module contains novel research contributions and experimental frameworks
for advancing the state-of-the-art in diffusion models for materials science.

Modules:
    benchmarking: Comprehensive benchmarking and comparison framework
"""

from .benchmarking import BenchmarkSuite, ExperimentConfig, BenchmarkResults, create_synthetic_dataset

__all__ = [
    'BenchmarkSuite',
    'ExperimentConfig', 
    'BenchmarkResults',
    'create_synthetic_dataset'
]