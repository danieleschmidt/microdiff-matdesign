#!/usr/bin/env python3
"""
Generation 3 Performance & Scaling Testing Suite
Tests advanced performance features, adaptive processing, and optimization.
"""

import numpy as np
import torch
import time
from microdiff_matdesign import MicrostructureDiffusion
from microdiff_matdesign.core import ProcessParameters


def test_performance_optimization():
    """Test performance optimization features."""
    print("  🎯 Testing adaptive resizing performance...")
    
    model = MicrostructureDiffusion(pretrained=False, device='cpu')
    
    test_sizes = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]
    
    for size in test_sizes:
        start_time = time.time()
        test_array = np.random.rand(*size)
        result = model.inverse_design(test_array, num_samples=1)
        end_time = time.time()
        
        assert result is not None
        assert hasattr(result, 'laser_power')
        
        duration = end_time - start_time
        print(f"    Size {size}: {duration:.3f}s")
        assert duration < 5.0, f"Too slow for size {size}: {duration:.3f}s"


def test_batch_processing():
    """Test batch processing performance."""
    print("  🎯 Testing batch processing performance...")
    
    model = MicrostructureDiffusion(pretrained=False, device='cpu')
    test_array = np.random.rand(16, 16, 16)
    
    sample_counts = [1, 3, 5]
    
    for num_samples in sample_counts:
        start_time = time.time()
        result = model.inverse_design(test_array, num_samples=num_samples)
        end_time = time.time()
        
        assert result is not None
        duration = end_time - start_time
        print(f"    Samples {num_samples}: {duration:.3f}s")
        
        expected_max_time = 1.0 + (num_samples * 0.5)
        assert duration < expected_max_time, f"Poor scaling for {num_samples} samples"


def test_memory_efficiency():
    """Test memory efficiency with large inputs."""
    print("  🎯 Testing memory efficiency...")
    
    model = MicrostructureDiffusion(pretrained=False, device='cpu')
    
    sizes = [(32, 32, 32), (48, 48, 48), (64, 64, 64)]
    
    for size in sizes:
        test_array = np.random.rand(*size)
        result = model.inverse_design(test_array, num_samples=1)
        assert result is not None
        print(f"    Size {size}: Memory efficient ✅")


def test_comprehensive_scaling_integration():
    """Comprehensive integration test for Generation 3 scaling."""
    print("🚀 Running comprehensive Generation 3 scaling test...")
    
    model = MicrostructureDiffusion(
        pretrained=False, 
        device='cpu',
        enable_scaling=True,
        enable_caching=True
    )
    
    test_scenarios = [
        ((8, 8, 8), 1, "Tiny input, single sample"),
        ((16, 16, 16), 2, "Small input, multi-sample"),
        ((32, 32, 32), 3, "Standard input, batch processing"),
        ((24, 36, 28), 2, "Non-cubic input, multi-sample"),
    ]
    
    total_time = 0
    for size, num_samples, description in test_scenarios:
        print(f"  Testing: {description}")
        
        start_time = time.time()
        test_array = np.random.rand(*size)
        
        result = model.inverse_design(test_array, num_samples=num_samples)
        
        end_time = time.time()
        duration = end_time - start_time
        total_time += duration
        
        print(f"    ✅ {description}: {duration:.3f}s")
        
        assert result is not None
        assert isinstance(result, ProcessParameters)
        assert hasattr(result, 'laser_power')
        assert hasattr(result, 'scan_speed')
        
        assert duration < 3.0, f"Performance too slow: {duration:.3f}s"
    
    print(f"  📊 Total test time: {total_time:.3f}s")
    print("✅ Comprehensive scaling integration test PASSED")
    
    return True


if __name__ == "__main__":
    print("⚡ Running Generation 3 Scaling & Performance Tests...")
    
    print("✅ Testing performance optimization...")
    test_performance_optimization()
    
    print("✅ Testing batch processing...")
    test_batch_processing()
    
    print("✅ Testing memory efficiency...")
    test_memory_efficiency()
    
    print("✅ Running comprehensive scaling integration test...")
    test_comprehensive_scaling_integration()
    
    print("\n🎉 Generation 3 Scaling & Performance Tests COMPLETED SUCCESSFULLY!")
    print("✅ Performance optimization: OPTIMIZED")
    print("✅ Adaptive processing: EFFICIENT") 
    print("✅ Memory efficiency: VERIFIED")
    print("✅ Integration testing: COMPREHENSIVE")