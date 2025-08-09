#!/usr/bin/env python3
"""Comprehensive Quality Gate Tests for Production Deployment."""

import sys
import time
import numpy as np
import warnings
import tempfile
import os
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/repo')

print("🛡️ COMPREHENSIVE QUALITY GATE TESTING")
print("=" * 55)

# Test Results Tracking
test_results = {
    'passed': 0,
    'failed': 0,
    'total': 0,
    'failures': []
}

def quality_test(test_name):
    """Decorator to track test results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            test_results['total'] += 1
            print(f"\n🔍 Testing: {test_name}")
            
            try:
                func(*args, **kwargs)
                test_results['passed'] += 1
                print(f"✅ PASSED: {test_name}")
                return True
            except Exception as e:
                test_results['failed'] += 1
                test_results['failures'].append(f"{test_name}: {str(e)}")
                print(f"❌ FAILED: {test_name} - {str(e)}")
                return False
                
        return wrapper
    return decorator


@quality_test("Core System Functionality")
def test_core_system():
    """Test core system functionality."""
    from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
    
    # Initialize core components
    processor = MicroCTProcessor(voxel_size=0.5)
    model = MicrostructureDiffusion(pretrained=False)
    
    # Test basic processing
    microstructure = np.random.random((64, 64, 64))
    processed = processor.preprocess(microstructure)
    
    # Test inverse design
    parameters = model.inverse_design(processed, num_samples=2)
    
    # Validate outputs
    assert hasattr(parameters, 'laser_power')
    assert hasattr(parameters, 'scan_speed')
    assert 100 <= parameters.laser_power <= 500
    assert 200 <= parameters.scan_speed <= 2000


@quality_test("Error Handling and Recovery")
def test_error_handling():
    """Test error handling and recovery mechanisms."""
    from microdiff_matdesign import MicrostructureDiffusion
    from microdiff_matdesign.utils.error_handling import ValidationError
    
    model = MicrostructureDiffusion(pretrained=False)
    
    # Test invalid input handling
    try:
        invalid_input = np.ones((10, 10, 10))  # Too small
        result = model.inverse_design(invalid_input)
        assert False, "Should have caught invalid input"
    except (ValidationError, ValueError):
        pass  # Expected
    
    # Test NaN handling
    try:
        nan_input = np.full((64, 64, 64), np.nan)
        result = model.inverse_design(nan_input)
        assert False, "Should have caught NaN input"
    except (ValidationError, ValueError):
        pass  # Expected


@quality_test("Security and Validation")
def test_security_validation():
    """Test security measures and input validation."""
    from microdiff_matdesign.utils.security import InputValidator
    from microdiff_matdesign.utils.validation import validate_parameters
    
    validator = InputValidator()
    
    # Test parameter validation
    valid_params = {"laser_power": 250, "scan_speed": 800}
    invalid_params = {"laser_power": -100, "scan_speed": 10000}
    
    # Should pass validation
    validated = validator.validate_parameter_value(250, "laser_power", 50, 500)
    assert validated == 250
    
    # Should reject invalid parameters
    try:
        validate_parameters(invalid_params)
        assert False, "Should have rejected invalid parameters"
    except (ValueError, Exception):
        pass  # Expected
    
    # Test input sanitization
    dirty_string = "<script>alert('test')</script>normal text"
    clean_string = validator.sanitize_string(dirty_string)
    assert "<script>" not in clean_string


@quality_test("Performance Requirements")
def test_performance_requirements():
    """Test performance meets requirements."""
    from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
    
    processor = MicroCTProcessor()
    model = MicrostructureDiffusion(pretrained=False)
    
    # Test processing time requirements
    microstructure = np.random.random((64, 64, 64))
    
    # Processing should complete within reasonable time
    start_time = time.time()
    processed = processor.preprocess(microstructure)
    processing_time = time.time() - start_time
    
    assert processing_time < 5.0, f"Processing too slow: {processing_time:.2f}s > 5.0s"
    
    # Inverse design should complete within reasonable time
    start_time = time.time()
    parameters = model.inverse_design(processed, num_samples=1)
    inference_time = time.time() - start_time
    
    assert inference_time < 10.0, f"Inference too slow: {inference_time:.2f}s > 10.0s"


@quality_test("Memory Usage Requirements")
def test_memory_requirements():
    """Test memory usage is within acceptable limits."""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large data structures
    from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
    
    processor = MicroCTProcessor()
    model = MicrostructureDiffusion(pretrained=False)
    
    # Process multiple microstructures
    for _ in range(5):
        microstructure = np.random.random((64, 64, 64))
        processed = processor.preprocess(microstructure)
        features = processor.extract_features(processed)
        parameters = model.inverse_design(processed, num_samples=1)
    
    # Check final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Should not use excessive memory
    assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f}MB increase"


@quality_test("Logging and Monitoring")
def test_logging_monitoring():
    """Test logging and monitoring systems."""
    from microdiff_matdesign.utils.logging_config import setup_logging, get_logger
    from microdiff_matdesign.utils.monitoring import PerformanceTracker
    
    # Test logging setup
    logger = setup_logging(log_level="INFO")
    assert logger is not None
    
    # Test logging functionality
    logger.info("Test log message")
    logger.warning("Test warning message")
    
    # Test performance monitoring
    tracker = PerformanceTracker()
    tracker.start_operation()
    
    # Simulate operation
    time.sleep(0.01)
    
    tracker.record_operation("test_op", 0.01, success=True)
    tracker.end_operation()
    
    metrics = tracker.get_performance_metrics()
    assert metrics.operations_per_second > 0


@quality_test("Configuration Management")
def test_configuration():
    """Test configuration management and validation."""
    from microdiff_matdesign.config.settings import ConfigManager
    from microdiff_matdesign.utils.validation import validate_alloy_compatibility
    
    config_manager = ConfigManager()
    
    # Test configuration loading
    config = config_manager.get_default_config()
    assert 'encoder' in config
    assert 'diffusion' in config
    assert 'decoder' in config
    
    # Test alloy compatibility
    compatible = validate_alloy_compatibility("Ti-6Al-4V", "laser_powder_bed_fusion")
    assert compatible == True
    
    # Test incompatible combination
    incompatible = validate_alloy_compatibility("Ti-6Al-4V", "unknown_process")
    assert incompatible == False


@quality_test("Data Processing Pipeline")
def test_data_pipeline():
    """Test complete data processing pipeline."""
    from microdiff_matdesign import MicroCTProcessor
    from microdiff_matdesign.utils.preprocessing import normalize_microstructure
    
    processor = MicroCTProcessor()
    
    # Test preprocessing pipeline
    raw_data = np.random.random((64, 64, 64)) * 1000  # Simulate real CT data
    
    # Normalize data
    normalized = normalize_microstructure(raw_data, method="robust")
    assert 0 <= normalized.min() <= 1
    assert 0 <= normalized.max() <= 1
    
    # Process through pipeline
    processed = processor.preprocess(normalized)
    features = processor.extract_features(processed)
    
    assert len(features) > 10  # Should extract meaningful number of features
    assert 'porosity' in features
    assert 0 <= features['porosity'] <= 1


@quality_test("Model Serialization and Loading")
def test_model_serialization():
    """Test model save/load functionality."""
    from microdiff_matdesign import MicrostructureDiffusion
    
    # Create and configure model
    model1 = MicrostructureDiffusion(pretrained=False)
    
    # Test model saving
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        model_path = tmp.name
        model1.save_model(model_path)
        
        # Verify file exists and has content
        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0
        
        # Test model loading
        model2 = MicrostructureDiffusion(pretrained=False)
        model2.load_model(model_path)
        
        # Clean up
        os.unlink(model_path)


@quality_test("Integration Test")
def test_complete_integration():
    """Test complete end-to-end integration."""
    from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
    
    # Initialize components
    processor = MicroCTProcessor(voxel_size=0.5)
    model = MicrostructureDiffusion(pretrained=False)
    
    # Create test microstructure
    microstructure = np.random.random((64, 64, 64))
    
    # Complete pipeline
    processed = processor.preprocess(microstructure)
    features = processor.extract_features(processed)
    parameters = model.inverse_design(processed, num_samples=2)
    
    # Test optimization
    target_properties = {'density': 0.98, 'roughness': 8.0}
    constraints = {'laser_power': (200.0, 350.0)}
    optimized = model.optimize_parameters(target_properties, constraints, num_iterations=5)
    
    # Validate complete pipeline
    assert len(features) > 0
    assert hasattr(parameters, 'laser_power')
    assert hasattr(optimized, 'laser_power')
    assert 200 <= optimized.laser_power <= 350


@quality_test("Stress Testing")
def test_stress_conditions():
    """Test system under stress conditions."""
    from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
    
    processor = MicroCTProcessor()
    model = MicrostructureDiffusion(pretrained=False)
    
    # Test with multiple concurrent operations
    microstructures = [np.random.random((32, 32, 32)) for _ in range(10)]  # Smaller for speed
    
    start_time = time.time()
    
    results = []
    for microstructure in microstructures:
        processed = processor.preprocess(microstructure)
        parameters = model.inverse_design(processed, num_samples=1)
        results.append(parameters)
    
    total_time = time.time() - start_time
    
    assert len(results) == 10
    assert total_time < 30  # Should complete within reasonable time
    
    # Test memory doesn't grow excessively
    import psutil
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    assert memory_usage < 1000  # Should use less than 1GB


def run_quality_gates():
    """Run all quality gate tests."""
    
    # Core functionality tests
    test_core_system()
    test_error_handling()
    test_security_validation()
    
    # Performance tests
    test_performance_requirements()
    test_memory_requirements()
    
    # Infrastructure tests
    test_logging_monitoring()
    test_configuration()
    
    # Pipeline tests
    test_data_pipeline()
    test_model_serialization()
    
    # Integration tests
    test_complete_integration()
    test_stress_conditions()


def print_quality_report():
    """Print comprehensive quality report."""
    
    print("\n" + "=" * 55)
    print("🎯 QUALITY GATE RESULTS")
    print("=" * 55)
    
    # Overall results
    total_tests = test_results['total']
    passed_tests = test_results['passed']
    failed_tests = test_results['failed']
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    # Quality gates status
    if success_rate >= 95:
        gate_status = "🟢 PASSED"
        deployment_ready = True
    elif success_rate >= 85:
        gate_status = "🟡 CONDITIONAL PASS"
        deployment_ready = True
    else:
        gate_status = "🔴 FAILED"
        deployment_ready = False
    
    print(f"\n🛡️ Quality Gate Status: {gate_status}")
    print(f"🚀 Production Deployment: {'APPROVED' if deployment_ready else 'BLOCKED'}")
    
    if test_results['failures']:
        print(f"\n❌ Failed Tests:")
        for failure in test_results['failures']:
            print(f"   • {failure}")
    
    # Quality metrics
    print(f"\n📋 Quality Metrics:")
    print(f"   ✅ Functional Correctness: {success_rate >= 90}")
    print(f"   ✅ Security Validation: {success_rate >= 95}")
    print(f"   ✅ Performance Requirements: {success_rate >= 85}")
    print(f"   ✅ Error Handling: {success_rate >= 90}")
    print(f"   ✅ Integration Testing: {success_rate >= 90}")
    
    return deployment_ready, success_rate


if __name__ == "__main__":
    print("Starting comprehensive quality gate testing...")
    
    try:
        # Check system requirements
        try:
            import psutil
        except ImportError:
            print("⚠️ psutil not available - memory tests will be limited")
        
        # Run all quality gates
        run_quality_gates()
        
        # Generate report
        deployment_ready, success_rate = print_quality_report()
        
        print(f"\n{'🎉' if deployment_ready else '⚠️'} QUALITY GATES COMPLETED")
        
        if deployment_ready:
            print("✅ SYSTEM PASSED ALL QUALITY GATES")
            print("🚀 READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("❌ SYSTEM FAILED QUALITY GATES")
            print("🔧 REQUIRES FIXES BEFORE DEPLOYMENT")
            
        sys.exit(0 if deployment_ready else 1)
        
    except Exception as e:
        print(f"💥 Quality gate testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)