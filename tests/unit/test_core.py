"""Unit tests for core functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch

from microdiff_matdesign.core import MicrostructureDiffusion


class TestMicrostructureDiffusion:
    """Test cases for MicrostructureDiffusion class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        model = MicrostructureDiffusion()
        assert model.alloy == "Ti-6Al-4V"
        assert model.process == "laser_powder_bed_fusion"
        assert not model.pretrained
    
    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        model = MicrostructureDiffusion(
            alloy="Inconel718",
            process="electron_beam_melting",
            pretrained=True
        )
        assert model.alloy == "Inconel718"
        assert model.process == "electron_beam_melting"
        assert model.pretrained
    
    @pytest.mark.parametrize("alloy", ["Ti-6Al-4V", "Inconel718", "AlSi10Mg"])
    def test_supported_alloys(self, alloy):
        """Test that supported alloys can be initialized."""
        model = MicrostructureDiffusion(alloy=alloy)
        assert model.alloy == alloy
    
    def test_unsupported_alloy_raises_error(self):
        """Test that unsupported alloys raise an error."""
        with pytest.raises(ValueError, match="Unsupported alloy"):
            MicrostructureDiffusion(alloy="UnsupportedAlloy")
    
    @pytest.mark.parametrize("process", [
        "laser_powder_bed_fusion",
        "electron_beam_melting", 
        "directed_energy_deposition"
    ])
    def test_supported_processes(self, process):
        """Test that supported processes can be initialized."""
        model = MicrostructureDiffusion(process=process)
        assert model.process == process
    
    def test_unsupported_process_raises_error(self):
        """Test that unsupported processes raise an error."""
        with pytest.raises(ValueError, match="Unsupported process"):
            MicrostructureDiffusion(process="unsupported_process")
    
    @patch('microdiff_matdesign.core.torch.load')
    def test_load_pretrained_model(self, mock_torch_load):
        """Test loading pretrained models."""
        mock_torch_load.return_value = {"model_state": "test"}
        
        model = MicrostructureDiffusion(pretrained=True)
        model.load_pretrained()
        
        mock_torch_load.assert_called_once()
    
    def test_inverse_design_input_validation(self, sample_microstructure):
        """Test input validation for inverse design."""
        model = MicrostructureDiffusion()
        
        # Test with valid input
        result = model.inverse_design(
            target_microstructure=sample_microstructure,
            num_samples=5
        )
        assert result is not None
        
        # Test with invalid microstructure
        with pytest.raises(ValueError, match="Target microstructure must be"):
            model.inverse_design(target_microstructure="invalid")
        
        # Test with invalid num_samples
        with pytest.raises(ValueError, match="num_samples must be positive"):
            model.inverse_design(
                target_microstructure=sample_microstructure,
                num_samples=0
            )
    
    @pytest.mark.slow
    def test_inverse_design_output_format(self, sample_microstructure):
        """Test that inverse design returns correct output format."""
        model = MicrostructureDiffusion()
        
        result = model.inverse_design(
            target_microstructure=sample_microstructure,
            num_samples=3
        )
        
        # Check output structure
        assert hasattr(result, 'laser_power')
        assert hasattr(result, 'scan_speed')
        assert hasattr(result, 'layer_thickness')
        assert hasattr(result, 'hatch_spacing')
        
        # Check value ranges (should be physically reasonable)
        assert 50 <= result.laser_power <= 500  # W
        assert 100 <= result.scan_speed <= 3000  # mm/s
        assert 10 <= result.layer_thickness <= 100  # μm
        assert 50 <= result.hatch_spacing <= 300  # μm
    
    @patch('microdiff_matdesign.core.DiffusionModel')
    def test_model_caching(self, mock_diffusion_model):
        """Test that models are cached properly."""
        model = MicrostructureDiffusion()
        
        # First call should create model
        model._get_model()
        assert mock_diffusion_model.called
        
        # Second call should use cached model
        mock_diffusion_model.reset_mock()
        model._get_model()
        assert not mock_diffusion_model.called
    
    def test_device_selection(self):
        """Test device selection logic."""
        model = MicrostructureDiffusion()
        
        # Should select appropriate device
        device = model._get_device()
        assert isinstance(device, torch.device)
        
        # Should handle CPU fallback
        with patch('torch.cuda.is_available', return_value=False):
            device = model._get_device()
            assert device.type == 'cpu'
    
    @pytest.mark.gpu
    def test_gpu_acceleration(self, sample_microstructure):
        """Test GPU acceleration when available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        model = MicrostructureDiffusion(device='cuda')
        result = model.inverse_design(
            target_microstructure=sample_microstructure,
            num_samples=1
        )
        assert result is not None
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        model = MicrostructureDiffusion()
        
        # Test valid parameters
        valid_params = {
            "laser_power": 250.0,
            "scan_speed": 1200.0,
            "layer_thickness": 30.0,
            "hatch_spacing": 120.0
        }
        assert model._validate_parameters(valid_params)
        
        # Test invalid parameters
        invalid_params = {
            "laser_power": -100.0,  # Negative power
            "scan_speed": 1200.0,
            "layer_thickness": 30.0,
            "hatch_spacing": 120.0
        }
        with pytest.raises(ValueError, match="Invalid parameter"):
            model._validate_parameters(invalid_params)
    
    def test_uncertainty_quantification(self, sample_microstructure):
        """Test uncertainty quantification functionality."""
        model = MicrostructureDiffusion()
        
        params, uncertainty = model.predict_with_uncertainty(
            target_microstructure=sample_microstructure,
            confidence_level=0.95
        )
        
        # Check that uncertainty bounds are returned
        assert isinstance(uncertainty, dict)
        for param_name in params.keys():
            assert param_name in uncertainty
            low, high = uncertainty[param_name]
            assert low <= params[param_name] <= high
    
    def test_batch_processing(self, sample_microstructure):
        """Test batch processing of multiple microstructures."""
        model = MicrostructureDiffusion()
        
        # Create batch of microstructures
        batch = np.stack([sample_microstructure] * 3)
        
        results = model.inverse_design_batch(batch)
        assert len(results) == 3
        
        for result in results:
            assert hasattr(result, 'laser_power')
            assert hasattr(result, 'scan_speed')
    
    def test_configuration_management(self):
        """Test configuration loading and saving."""
        model = MicrostructureDiffusion()
        
        # Test default configuration
        config = model.get_config()
        assert 'alloy' in config
        assert 'process' in config
        
        # Test custom configuration
        custom_config = {
            'alloy': 'Inconel718',
            'process': 'electron_beam_melting',
            'model_params': {'diffusion_steps': 500}
        }
        model.update_config(custom_config)
        
        updated_config = model.get_config()
        assert updated_config['alloy'] == 'Inconel718'
        assert updated_config['model_params']['diffusion_steps'] == 500
    
    @pytest.mark.benchmark
    def test_inference_performance(self, benchmark, sample_microstructure):
        """Benchmark inference performance."""
        model = MicrostructureDiffusion()
        
        def run_inference():
            return model.inverse_design(
                target_microstructure=sample_microstructure,
                num_samples=1
            )
        
        result = benchmark(run_inference)
        assert result is not None
    
    def test_memory_management(self, sample_microstructure):
        """Test memory management for large volumes."""
        # Create a larger volume to test memory handling
        large_volume = np.random.randint(0, 3, (256, 256, 256), dtype=np.uint8)
        
        model = MicrostructureDiffusion()
        
        # Should handle large volumes without memory errors
        result = model.inverse_design(
            target_microstructure=large_volume,
            num_samples=1
        )
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling for various failure modes."""
        model = MicrostructureDiffusion()
        
        # Test with corrupted microstructure
        corrupted_volume = np.full((64, 64, 64), np.nan)
        with pytest.raises(ValueError, match="contains invalid values"):
            model.inverse_design(target_microstructure=corrupted_volume)
        
        # Test with wrong dimensions
        wrong_dims = np.random.rand(64, 64)  # 2D instead of 3D
        with pytest.raises(ValueError, match="must be 3D"):
            model.inverse_design(target_microstructure=wrong_dims)
    
    def test_reproducibility(self, sample_microstructure):
        """Test that results are reproducible with same seed."""
        model = MicrostructureDiffusion(random_seed=42)
        
        result1 = model.inverse_design(
            target_microstructure=sample_microstructure,
            num_samples=1
        )
        
        model_2 = MicrostructureDiffusion(random_seed=42)
        result2 = model_2.inverse_design(
            target_microstructure=sample_microstructure, 
            num_samples=1
        )
        
        # Results should be identical with same seed
        assert abs(result1.laser_power - result2.laser_power) < 1e-6
        assert abs(result1.scan_speed - result2.scan_speed) < 1e-6