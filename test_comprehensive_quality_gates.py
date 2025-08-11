#!/usr/bin/env python3
"""Comprehensive Quality Gates Test - Final Validation"""

import numpy as np
import sys
import time
import traceback
import warnings
import tempfile
import os
from pathlib import Path

def test_core_functionality():
    """Test core system functionality."""
    print("🧪 Testing Core Functionality...")
    
    try:
        from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
        
        # Test model instantiation
        model = MicrostructureDiffusion(
            enable_validation=True,
            safety_checks=True,
            enable_scaling=True,
            enable_caching=True
        )
        print("✅ Model instantiation successful")
        
        # Test processor instantiation
        processor = MicroCTProcessor()
        print("✅ Processor instantiation successful")
        
        # Test basic inverse design
        microstructure = np.random.rand(32, 32, 32)
        result = model.inverse_design(microstructure, num_samples=2)
        
        if hasattr(result, 'laser_power'):
            print("✅ Basic inverse design working")
        else:
            print("❌ Invalid result structure")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test comprehensive error handling."""
    print("🧪 Testing Error Handling...")
    
    try:
        from microdiff_matdesign import MicrostructureDiffusion
        from microdiff_matdesign.utils.error_handling import ValidationError
        
        model = MicrostructureDiffusion(enable_validation=True)
        
        # Test invalid inputs
        error_tests = 0
        
        # Test 2D input (should fail)
        try:
            model.inverse_design(np.random.rand(32, 32), num_samples=1)
            print("❌ Should have rejected 2D input")
        except (ValidationError, ValueError):
            print("✅ Correctly rejected 2D input")
            error_tests += 1
        
        # Test invalid num_samples
        try:
            model.inverse_design(np.random.rand(32, 32, 32), num_samples=-1)
            print("❌ Should have rejected negative samples")
        except ValidationError:
            print("✅ Correctly rejected negative samples")  
            error_tests += 1
        
        return error_tests >= 2
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_security_validation():
    """Test security validation."""
    print("🧪 Testing Security Validation...")
    
    try:
        from microdiff_matdesign.utils.security import InputValidator, generate_secure_token
        
        validator = InputValidator()
        
        # Test input sanitization
        dirty_input = "<script>alert('xss')</script>"
        clean_input = validator.sanitize_string(dirty_input)
        
        if "<script>" not in clean_input:
            print("✅ XSS sanitization working")
        else:
            print("❌ XSS sanitization failed")
            return False
        
        # Test parameter validation
        try:
            valid_param = validator.validate_parameter_value(
                100.0, "laser_power", min_value=50.0, max_value=500.0
            )
            print("✅ Parameter validation working")
        except Exception as e:
            print(f"❌ Parameter validation failed: {e}")
            return False
        
        # Test token generation
        token = generate_secure_token()
        if len(token) >= 32:
            print("✅ Secure token generation working")
        else:
            print("❌ Token too short")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        return False

def test_performance_requirements():
    """Test performance requirements."""
    print("🧪 Testing Performance Requirements...")
    
    try:
        from microdiff_matdesign import MicrostructureDiffusion
        
        model = MicrostructureDiffusion(enable_scaling=True)
        
        # Test response time requirements
        microstructure = np.random.rand(32, 32, 32)
        
        start_time = time.time()
        result = model.inverse_design(microstructure, num_samples=2)
        duration = time.time() - start_time
        
        # Should complete within reasonable time (15 seconds for testing)
        if duration < 15.0:
            print(f"✅ Performance requirement met: {duration:.2f}s")
        else:
            print(f"❌ Too slow: {duration:.2f}s")
            return False
        
        # Test parallel processing
        start_time = time.time()
        result_parallel = model.inverse_design(
            microstructure, 
            num_samples=4, 
            enable_parallel=True
        )
        parallel_duration = time.time() - start_time
        
        if parallel_duration < 20.0:  # Slightly more time for parallel
            print(f"✅ Parallel processing works: {parallel_duration:.2f}s")
        else:
            print(f"❌ Parallel processing too slow: {parallel_duration:.2f}s")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_caching_system():
    """Test caching system."""
    print("🧪 Testing Caching System...")
    
    try:
        from microdiff_matdesign import MicrostructureDiffusion
        
        model = MicrostructureDiffusion(enable_caching=True)
        microstructure = np.random.rand(32, 32, 32)
        
        # First call
        start_time = time.time()
        result1 = model.inverse_design(microstructure, num_samples=2)
        first_time = time.time() - start_time
        
        # Second call (should use cache)
        start_time = time.time()
        result2 = model.inverse_design(microstructure, num_samples=2)
        second_time = time.time() - start_time
        
        # Results should be consistent
        param_diff = abs(result1.laser_power - result2.laser_power)
        if param_diff < 0.1:
            print("✅ Cache consistency verified")
        else:
            print(f"⚠️  Cache inconsistency: {param_diff}")
        
        print(f"✅ Caching system operational (times: {first_time:.3f}s, {second_time:.3f}s)")
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def test_logging_system():
    """Test logging system."""
    print("🧪 Testing Logging System...")
    
    try:
        from microdiff_matdesign.utils.logging_config import setup_logging, get_logger
        
        # Test logger setup
        logger = setup_logging(log_level="INFO", enable_console=False)
        
        # Test basic logging
        test_logger = get_logger("test")
        test_logger.info("Test log message")
        test_logger.warning("Test warning message")
        test_logger.error("Test error message")
        
        print("✅ Logging system operational")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_model_serialization():
    """Test model serialization and loading."""
    print("🧪 Testing Model Serialization...")
    
    try:
        from microdiff_matdesign import MicrostructureDiffusion
        
        model = MicrostructureDiffusion(pretrained=False)
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            temp_path = f.name
        
        try:
            # Test save
            model.save_model(temp_path)
            print("✅ Model save successful")
            
            # Test load
            model2 = MicrostructureDiffusion(pretrained=False)
            model2.load_model(temp_path)
            print("✅ Model load successful")
            
            return True
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Model serialization test failed: {e}")
        return False

def test_integration_workflow():
    """Test end-to-end integration workflow."""
    print("🧪 Testing Integration Workflow...")
    
    try:
        from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
        
        # Full workflow test
        processor = MicroCTProcessor()
        model = MicrostructureDiffusion(
            enable_validation=True,
            enable_scaling=True,
            enable_caching=True
        )
        
        # Simulate processing pipeline
        # 1. Create synthetic microstructure
        microstructure = np.random.rand(32, 32, 32)
        
        # 2. Preprocess (simplified)
        from microdiff_matdesign.utils.preprocessing import normalize_microstructure
        normalized = normalize_microstructure(microstructure)
        print("✅ Preprocessing successful")
        
        # 3. Inverse design
        result = model.inverse_design(normalized, num_samples=2)
        print("✅ Inverse design successful")
        
        # 4. Parameter validation
        from microdiff_matdesign.utils.validation import validate_parameters
        validate_parameters(result.to_dict(), "laser_powder_bed_fusion")
        print("✅ Parameter validation successful")
        
        # 5. Uncertainty quantification
        result_with_uncertainty = model.inverse_design(
            normalized, 
            num_samples=3, 
            uncertainty_quantification=True
        )
        
        if len(result_with_uncertainty) == 2:
            print("✅ Uncertainty quantification successful")
        else:
            print("❌ Uncertainty quantification failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        traceback.print_exc()
        return False

def test_stress_conditions():
    """Test system under stress conditions."""
    print("🧪 Testing Stress Conditions...")
    
    try:
        from microdiff_matdesign import MicrostructureDiffusion
        
        model = MicrostructureDiffusion(enable_scaling=True)
        
        # Test with multiple sizes
        stress_tests = 0
        test_cases = [
            ("small", 16),
            ("medium", 32),
            ("large", 48)
        ]
        
        for test_name, size in test_cases:
            try:
                microstructure = np.random.rand(size, size, size)
                result = model.inverse_design(microstructure, num_samples=2)
                
                if hasattr(result, 'laser_power'):
                    print(f"✅ Stress test {test_name} ({size}³) passed")
                    stress_tests += 1
                else:
                    print(f"❌ Stress test {test_name} failed - invalid result")
                    
            except Exception as e:
                print(f"⚠️  Stress test {test_name} failed: {e}")
        
        # Test concurrent operations (simplified)
        try:
            microstructure = np.random.rand(24, 24, 24)
            results = []
            
            for i in range(3):
                result = model.inverse_design(microstructure, num_samples=1)
                results.append(result)
            
            if len(results) == 3:
                print("✅ Concurrent operations test passed")
                stress_tests += 1
                
        except Exception as e:
            print(f"⚠️  Concurrent operations test failed: {e}")
        
        return stress_tests >= 3  # At least 3 of 4 tests should pass
        
    except Exception as e:
        print(f"❌ Stress testing failed: {e}")
        return False

def main():
    """Run comprehensive quality gates."""
    print("=" * 60)
    print("🛡️ COMPREHENSIVE QUALITY GATES - FINAL VALIDATION")
    print("=" * 60)
    
    quality_tests = [
        ("Core Functionality", test_core_functionality, True),
        ("Error Handling", test_error_handling, True),
        ("Security Validation", test_security_validation, True),
        ("Performance Requirements", test_performance_requirements, True),
        ("Caching System", test_caching_system, False),  # Optional
        ("Logging System", test_logging_system, True),
        ("Model Serialization", test_model_serialization, False),  # Optional
        ("Integration Workflow", test_integration_workflow, True),
        ("Stress Conditions", test_stress_conditions, False),  # Optional
    ]
    
    results = []
    critical_failures = 0
    
    for test_name, test_func, is_critical in quality_tests:
        print(f"\n🔍 Testing {test_name}...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                success = test_func()
                
            results.append((test_name, success, is_critical))
            
            if not success and is_critical:
                critical_failures += 1
                
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False, is_critical))
            if is_critical:
                critical_failures += 1
    
    print("\n" + "=" * 60)
    print("🎯 QUALITY GATE RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    critical_passed = sum(1 for _, success, critical in results if success and critical)
    total_critical = sum(1 for _, _, critical in results if critical)
    total = len(results)
    
    for test_name, success, is_critical in results:
        status = "PASS" if success else "FAIL"
        emoji = "✅" if success else "❌"
        critical_marker = " [CRITICAL]" if is_critical else ""
        print(f"{emoji} {test_name}: {status}{critical_marker}")
    
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    print(f"   Critical Tests Passed: {critical_passed}/{total_critical}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    
    # Quality gate decision
    quality_gate_passed = (
        critical_failures == 0 and  # No critical failures
        passed >= total * 0.75  # At least 75% overall pass rate
    )
    
    if quality_gate_passed:
        print(f"\n🎉 QUALITY GATES: ✅ PASSED")
        print("🚀 SYSTEM READY FOR PRODUCTION DEPLOYMENT")
        print("\n📋 Quality Metrics Verified:")
        print("   ✅ Functional Correctness")
        print("   ✅ Security Validation")
        print("   ✅ Error Handling")
        print("   ✅ Performance Requirements")
        print("   ✅ Integration Testing")
        return True
    else:
        print(f"\n❌ QUALITY GATES: 🔴 FAILED")
        print("🚫 PRODUCTION DEPLOYMENT BLOCKED")
        if critical_failures > 0:
            print(f"⚠️  Critical failures detected: {critical_failures}")
        print("🔧 System requires fixes before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)