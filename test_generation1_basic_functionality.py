#!/usr/bin/env python3
"""Generation 1 Basic Functionality Test - MAKE IT WORK"""

import numpy as np
import sys
import traceback

def test_basic_import():
    """Test that the package imports successfully."""
    try:
        import microdiff_matdesign
        print(f"✅ Package import successful: {microdiff_matdesign.__version__}")
        return True
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return False

def test_core_classes():
    """Test that core classes can be instantiated."""
    try:
        from microdiff_matdesign import MicrostructureDiffusion, MicroCTProcessor
        
        # Test MicrostructureDiffusion
        model = MicrostructureDiffusion(pretrained=False)
        print("✅ MicrostructureDiffusion instantiated successfully")
        
        # Test MicroCTProcessor  
        processor = MicroCTProcessor()
        print("✅ MicroCTProcessor instantiated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Core class instantiation failed: {e}")
        traceback.print_exc()
        return False

def test_basic_inverse_design():
    """Test basic inverse design functionality."""
    try:
        from microdiff_matdesign import MicrostructureDiffusion
        
        # Create model
        model = MicrostructureDiffusion(pretrained=False)
        
        # Create dummy microstructure (64x64x64 for Generation 1)
        dummy_microstructure = np.random.rand(64, 64, 64).astype(np.float32)
        
        # Test inverse design
        parameters = model.inverse_design(
            target_microstructure=dummy_microstructure,
            num_samples=2,  # Keep small for testing
            guidance_scale=1.0  # Simple for Generation 1
        )
        
        print(f"✅ Inverse design successful - Laser power: {parameters.laser_power}W")
        print(f"   Parameters: {parameters.to_dict()}")
        
        return True
    except Exception as e:
        print(f"❌ Inverse design failed: {e}")
        traceback.print_exc()
        return False

def test_parameter_optimization():
    """Test basic parameter optimization."""
    try:
        from microdiff_matdesign import MicrostructureDiffusion
        
        model = MicrostructureDiffusion(pretrained=False)
        
        # Define target properties
        target_props = {
            'density': 0.98,
            'strength': 1100
        }
        
        # Test optimization
        optimized_params = model.optimize_parameters(
            target_properties=target_props,
            num_iterations=5  # Keep small for Generation 1
        )
        
        print(f"✅ Parameter optimization successful")
        print(f"   Optimized: {optimized_params.to_dict()}")
        
        return True
    except Exception as e:
        print(f"❌ Parameter optimization failed: {e}")
        traceback.print_exc()
        return False

def test_microct_processing():
    """Test basic micro-CT processing."""
    try:
        from microdiff_matdesign import MicroCTProcessor
        
        processor = MicroCTProcessor(voxel_size=0.5)
        
        # Create dummy 3D volume
        dummy_volume = np.random.randint(0, 255, (32, 32, 32)).astype(np.uint8)
        
        # Test preprocessing
        from microdiff_matdesign.utils.preprocessing import normalize_microstructure
        normalized = normalize_microstructure(dummy_volume.astype(np.float32))
        
        print(f"✅ Micro-CT processing successful - Shape: {normalized.shape}")
        print(f"   Normalized range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        
        return True
    except Exception as e:
        print(f"❌ Micro-CT processing failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 1 basic functionality tests."""
    print("=" * 60)
    print("GENERATION 1 BASIC FUNCTIONALITY TEST - MAKE IT WORK")
    print("=" * 60)
    
    tests = [
        ("Package Import", test_basic_import),
        ("Core Classes", test_core_classes), 
        ("Inverse Design", test_basic_inverse_design),
        ("Parameter Optimization", test_parameter_optimization),
        ("Micro-CT Processing", test_microct_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("GENERATION 1 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        emoji = "✅" if success else "❌"
        print(f"{emoji} {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 GENERATION 1 COMPLETE - Basic functionality working!")
        return True
    else:
        print("⚠️  Some tests failed - needs debugging")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)