#!/usr/bin/env python3
"""
Standalone test for basic MicroDiff functionality without heavy dependencies.
"""

import sys
import os
import numpy as np

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports work."""
    try:
        # Mock torch to avoid dependency issues
        import types
        torch_mock = types.ModuleType('torch')
        torch_mock.device = lambda x: f"device('{x}')"
        torch_mock.cuda = types.ModuleType('cuda')
        torch_mock.cuda.is_available = lambda: False
        torch_mock.tensor = lambda x, **kwargs: np.array(x)
        torch_mock.from_numpy = lambda x: x
        torch_mock.randn = lambda *args, **kwargs: np.random.randn(*args)
        torch_mock.no_grad = lambda: types.SimpleNamespace(__enter__=lambda: None, __exit__=lambda *args: None)
        
        # Add nn submodule
        torch_mock.nn = types.ModuleType('nn')
        torch_mock.nn.Module = object
        torch_mock.nn.functional = types.ModuleType('functional')
        torch_mock.nn.functional.interpolate = lambda x, **kwargs: x
        torch_mock.optim = types.ModuleType('optim')
        torch_mock.optim.AdamW = object
        torch_mock.optim.lr_scheduler = types.ModuleType('lr_scheduler')
        torch_mock.optim.lr_scheduler.CosineAnnealingLR = object
        
        sys.modules['torch'] = torch_mock
        sys.modules['torch.nn'] = torch_mock.nn
        sys.modules['torch.nn.functional'] = torch_mock.nn.functional
        sys.modules['torch.optim'] = torch_mock.optim
        sys.modules['torch.optim.lr_scheduler'] = torch_mock.optim.lr_scheduler
        
        # Test imports
        from microdiff_matdesign.core import ProcessParameters, MicrostructureDiffusion
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_process_parameters():
    """Test ProcessParameters class."""
    try:
        from microdiff_matdesign.core import ProcessParameters
        
        # Test default initialization
        params = ProcessParameters()
        assert params.laser_power == 200.0
        assert params.scan_speed == 800.0
        print("✅ ProcessParameters default initialization works")
        
        # Test custom initialization
        params = ProcessParameters(laser_power=250.0, scan_speed=900.0)
        assert params.laser_power == 250.0
        assert params.scan_speed == 900.0
        print("✅ ProcessParameters custom initialization works")
        
        # Test to_dict method
        param_dict = params.to_dict()
        assert isinstance(param_dict, dict)
        assert 'laser_power' in param_dict
        print("✅ ProcessParameters to_dict() works")
        
        return True
    except Exception as e:
        print(f"❌ ProcessParameters test failed: {e}")
        return False

def test_microstructure_diffusion_init():
    """Test MicrostructureDiffusion initialization."""
    try:
        # Mock additional dependencies
        import types
        
        # Mock yaml
        yaml_mock = types.ModuleType('yaml')
        yaml_mock.safe_load = lambda x: {
            'encoder': {'input_dim': 32768, 'hidden_dim': 256, 'latent_dim': 128},
            'diffusion': {'input_dim': 128, 'hidden_dim': 256, 'num_steps': 10},
            'decoder': {'latent_dim': 128, 'hidden_dim': 256, 'output_dim': 5}
        }
        sys.modules['yaml'] = yaml_mock
        
        # Mock tqdm
        tqdm_mock = types.ModuleType('tqdm')
        tqdm_mock.tqdm = lambda x, **kwargs: x
        sys.modules['tqdm'] = tqdm_mock
        
        # Mock model components
        def mock_model_init(self, *args, **kwargs):
            pass
        def mock_to(self, device):
            return self
        def mock_eval(self):
            pass
        def mock_train(self):
            pass
            
        mock_encoder = type('MicrostructureEncoder', (), {
            '__init__': mock_model_init,
            'to': mock_to,
            'eval': mock_eval,
            'train': mock_train
        })
        mock_diffusion = type('DiffusionModel', (), {
            '__init__': mock_model_init,
            'to': mock_to,
            'eval': mock_eval,
            'train': mock_train
        })
        mock_decoder = type('ParameterDecoder', (), {
            '__init__': mock_model_init,
            'to': mock_to,
            'eval': mock_eval,
            'train': mock_train
        })
        
        # Mock model modules
        models_mock = types.ModuleType('models')
        models_mock.encoders = types.ModuleType('encoders')
        models_mock.encoders.MicrostructureEncoder = mock_encoder
        models_mock.diffusion = types.ModuleType('diffusion')
        models_mock.diffusion.DiffusionModel = mock_diffusion
        models_mock.decoders = types.ModuleType('decoders')
        models_mock.decoders.ParameterDecoder = mock_decoder
        
        sys.modules['microdiff_matdesign.models'] = models_mock
        sys.modules['microdiff_matdesign.models.encoders'] = models_mock.encoders
        sys.modules['microdiff_matdesign.models.diffusion'] = models_mock.diffusion
        sys.modules['microdiff_matdesign.models.decoders'] = models_mock.decoders
        
        # Mock utils modules
        utils_mock = types.ModuleType('utils')
        utils_mock.validation = types.ModuleType('validation')
        utils_mock.validation.validate_microstructure = lambda x: True
        utils_mock.validation.validate_parameters = lambda x, y: True
        utils_mock.preprocessing = types.ModuleType('preprocessing')
        utils_mock.preprocessing.normalize_microstructure = lambda x: x
        utils_mock.preprocessing.denormalize_parameters = lambda x: x
        utils_mock.error_handling = types.ModuleType('error_handling')
        utils_mock.error_handling.handle_errors = lambda **kwargs: lambda func: func
        utils_mock.error_handling.error_context = lambda x: types.SimpleNamespace(__enter__=lambda: None, __exit__=lambda *args: None)
        utils_mock.error_handling.validate_input = lambda condition, message, exception_type: None
        utils_mock.error_handling.ValidationError = Exception
        utils_mock.error_handling.ModelError = Exception
        utils_mock.error_handling.ProcessingError = Exception
        utils_mock.logging_config = types.ModuleType('logging_config')
        utils_mock.logging_config.get_logger = lambda x: types.SimpleNamespace(info=print, warning=print, error=print)
        utils_mock.logging_config.with_logging = lambda x: lambda func: func
        utils_mock.logging_config.LoggingContextManager = lambda logger, context: types.SimpleNamespace(__enter__=lambda: None, __exit__=lambda *args: None)
        utils_mock.robust_validation = types.ModuleType('robust_validation')
        utils_mock.robust_validation.RobustValidator = lambda: None
        
        sys.modules['microdiff_matdesign.utils'] = utils_mock
        sys.modules['microdiff_matdesign.utils.validation'] = utils_mock.validation
        sys.modules['microdiff_matdesign.utils.preprocessing'] = utils_mock.preprocessing
        sys.modules['microdiff_matdesign.utils.error_handling'] = utils_mock.error_handling
        sys.modules['microdiff_matdesign.utils.logging_config'] = utils_mock.logging_config
        sys.modules['microdiff_matdesign.utils.robust_validation'] = utils_mock.robust_validation
        
        from microdiff_matdesign.core import MicrostructureDiffusion
        
        # Test initialization with minimal dependencies
        model = MicrostructureDiffusion(
            alloy="Ti-6Al-4V",
            process="laser_powder_bed_fusion",
            pretrained=False,
            enable_validation=False,
            enable_scaling=False,
            enable_caching=False
        )
        
        assert model.alloy == "Ti-6Al-4V"
        assert model.process == "laser_powder_bed_fusion"
        print("✅ MicrostructureDiffusion initialization works")
        
        return True
    except Exception as e:
        print(f"❌ MicrostructureDiffusion initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("🚀 Running standalone basic functionality tests...")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_process_parameters,
        test_microstructure_diffusion_init
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"💥 {test.__name__} failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests PASSED!")
        return 0
    else:
        print("💥 Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())