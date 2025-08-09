#!/usr/bin/env python3
"""Simplified test for basic functionality."""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== Testing Basic Components ===")

try:
    import sys
    sys.path.insert(0, '/root/repo')
    
    # Test basic imports
    from microdiff_matdesign.core import ProcessParameters
    print("✓ ProcessParameters imported")
    
    # Test parameter creation
    params = ProcessParameters(laser_power=250, scan_speed=900)
    print(f"✓ Created parameters: {params.laser_power}W, {params.scan_speed}mm/s")
    
    # Test tensor conversion
    import torch
    tensor = params.to_tensor()
    print(f"✓ Parameter tensor shape: {tensor.shape}")
    
    print("\n=== Basic Tests Passed ===")
    print("Ready for more comprehensive testing")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()