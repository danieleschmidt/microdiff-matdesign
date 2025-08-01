"""Integration tests for end-to-end pipelines."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from microdiff_matdesign.core import MicrostructureDiffusion
from microdiff_matdesign.imaging import MicroCTProcessor


class TestInverseDesignPipeline:
    """Integration tests for complete inverse design pipeline."""
    
    @pytest.mark.integration
    def test_complete_pipeline(self, sample_microstructure, temp_dir):
        """Test complete pipeline from microstructure to parameters."""
        # Initialize components
        model = MicrostructureDiffusion(alloy="Ti-6Al-4V")
        processor = MicroCTProcessor()
        
        # Save sample microstructure
        microstructure_path = temp_dir / "test_microstructure.npy"
        np.save(microstructure_path, sample_microstructure)
        
        # Load and process
        loaded_volume = processor.load_volume(str(microstructure_path))
        processed_volume = processor.preprocess(loaded_volume)
        
        # Run inverse design
        parameters = model.inverse_design(
            target_microstructure=processed_volume,
            num_samples=5
        )
        
        # Validate results
        assert parameters is not None
        assert hasattr(parameters, 'laser_power')
        assert hasattr(parameters, 'scan_speed')
        assert 50 <= parameters.laser_power <= 500
        assert 100 <= parameters.scan_speed <= 3000
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_alloy_pipeline(self, sample_microstructure):
        """Test pipeline with different alloys."""
        alloys = ["Ti-6Al-4V", "Inconel718", "AlSi10Mg"]
        results = {}
        
        for alloy in alloys:
            model = MicrostructureDiffusion(alloy=alloy)
            parameters = model.inverse_design(
                target_microstructure=sample_microstructure,
                num_samples=1
            )
            results[alloy] = parameters
        
        # Results should vary by alloy
        assert len(results) == 3
        for alloy, params in results.items():
            assert params is not None
            assert hasattr(params, 'laser_power')
    
    @pytest.mark.integration
    def test_multi_process_pipeline(self, sample_microstructure):
        """Test pipeline with different manufacturing processes."""
        processes = [
            "laser_powder_bed_fusion",
            "electron_beam_melting", 
            "directed_energy_deposition"
        ]
        results = {}
        
        for process in processes:
            model = MicrostructureDiffusion(process=process)
            parameters = model.inverse_design(
                target_microstructure=sample_microstructure,
                num_samples=1
            )
            results[process] = parameters
        
        # Results should vary by process
        assert len(results) == 3
        for process, params in results.items():
            assert params is not None
    
    @pytest.mark.integration
    def test_uncertainty_pipeline(self, sample_microstructure):
        """Test uncertainty quantification pipeline."""
        model = MicrostructureDiffusion()
        
        # Get parameters with uncertainty
        params, uncertainty = model.predict_with_uncertainty(
            target_microstructure=sample_microstructure,
            confidence_level=0.95,
            num_samples=10
        )
        
        # Validate uncertainty quantification
        assert isinstance(uncertainty, dict)
        for param_name, value in params.items():
            assert param_name in uncertainty
            low, high = uncertainty[param_name]
            assert low <= value <= high
            assert high > low  # Should have non-zero uncertainty
    
    @pytest.mark.integration
    def test_optimization_pipeline(self, sample_microstructure):
        """Test multi-objective optimization pipeline."""
        model = MicrostructureDiffusion()
        
        # Define objectives
        objectives = {
            "minimize_cost": True,
            "maximize_density": True,
            "minimize_surface_roughness": True
        }
        
        # Define constraints
        constraints = {
            "min_tensile_strength": 1000,  # MPa
            "max_porosity": 0.02,  # 2%
            "manufacturability": True
        }
        
        # Run optimization
        optimized_params = model.optimize_parameters(
            target_microstructure=sample_microstructure,
            objectives=objectives,
            constraints=constraints,
            num_generations=10
        )
        
        assert optimized_params is not None
        assert hasattr(optimized_params, 'laser_power')
        
        # Validate constraint satisfaction
        assert model._validate_constraints(optimized_params, constraints)
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_real_data_pipeline(self, test_data_dir):
        """Test pipeline with real microstructure data."""
        # Skip if no real data available
        real_data_path = test_data_dir / "real_microstructure.npy"
        if not real_data_path.exists():
            pytest.skip("Real microstructure data not available")
        
        # Load real data
        real_microstructure = np.load(real_data_path)
        
        # Process through pipeline
        model = MicrostructureDiffusion(alloy="Ti-6Al-4V")
        processor = MicroCTProcessor()
        
        processed = processor.preprocess(real_microstructure)
        parameters = model.inverse_design(
            target_microstructure=processed,
            num_samples=3
        )
        
        # Validate realistic parameter ranges
        assert 100 <= parameters.laser_power <= 400
        assert 500 <= parameters.scan_speed <= 2000
        assert 20 <= parameters.layer_thickness <= 60
    
    @pytest.mark.integration
    def test_batch_processing_pipeline(self, sample_microstructure):
        """Test batch processing pipeline."""
        # Create batch of different sizes
        batch_microstructures = [
            sample_microstructure,
            sample_microstructure[:32, :32, :32],  # Smaller
            np.tile(sample_microstructure, (2, 1, 1))[:64, :64, :64]  # Different
        ]
        
        model = MicrostructureDiffusion()
        processor = MicroCTProcessor()
        
        # Process batch
        processed_batch = []
        for microstructure in batch_microstructures:
            processed = processor.preprocess(microstructure)
            processed_batch.append(processed)
        
        # Run batch inference
        results = model.inverse_design_batch(processed_batch)
        
        assert len(results) == len(batch_microstructures)
        for result in results:
            assert result is not None
            assert hasattr(result, 'laser_power')
    
    @pytest.mark.integration 
    @pytest.mark.slow
    def test_training_pipeline(self, temp_dir, sample_microstructure, sample_parameters):
        """Test model training pipeline."""
        # Create synthetic training data
        training_data = []
        for i in range(10):  # Small dataset for testing
            # Generate slightly different microstructures
            noise = np.random.normal(0, 0.1, sample_microstructure.shape)
            noisy_microstructure = sample_microstructure + noise
            
            # Generate corresponding parameters with some variation
            varied_params = sample_parameters.copy()
            for key in varied_params:
                varied_params[key] *= (1 + np.random.normal(0, 0.1))
            
            training_data.append((noisy_microstructure, varied_params))
        
        # Initialize model for training
        model = MicrostructureDiffusion(alloy="Ti-6Al-4V")
        
        # Configure training
        training_config = {
            "epochs": 2,  # Very short for testing
            "batch_size": 2,
            "learning_rate": 1e-4,
            "validation_split": 0.2,
            "checkpoint_dir": str(temp_dir / "checkpoints")
        }
        
        # Run training
        training_history = model.train(
            training_data=training_data,
            config=training_config
        )
        
        # Validate training completed
        assert training_history is not None
        assert 'loss' in training_history
        assert len(training_history['loss']) == training_config['epochs']
    
    @pytest.mark.integration
    def test_validation_pipeline(self, sample_microstructure):
        """Test model validation pipeline."""
        model = MicrostructureDiffusion()
        
        # Create validation dataset
        validation_data = []
        for i in range(5):
            microstructure = sample_microstructure + np.random.normal(0, 0.05, sample_microstructure.shape)
            parameters = {
                "laser_power": 250 + np.random.normal(0, 25),
                "scan_speed": 1200 + np.random.normal(0, 120),
                "layer_thickness": 30 + np.random.normal(0, 3),
                "hatch_spacing": 120 + np.random.normal(0, 12)
            }
            validation_data.append((microstructure, parameters))
        
        # Run validation
        metrics = model.validate(validation_data)
        
        # Check metrics
        assert 'mae' in metrics  # Mean Absolute Error
        assert 'rmse' in metrics  # Root Mean Square Error
        assert 'r2' in metrics  # R-squared
        assert 'ssim' in metrics  # Structural Similarity (for microstructures)
        
        # Metrics should be reasonable
        assert 0 <= metrics['r2'] <= 1
        assert 0 <= metrics['ssim'] <= 1
    
    @pytest.mark.integration
    def test_configuration_pipeline(self, temp_dir):
        """Test configuration management pipeline."""
        config_path = temp_dir / "test_config.yaml"
        
        # Create configuration
        config = {
            "model": {
                "architecture": "unet3d",
                "diffusion_steps": 1000,
                "guidance_scale": 7.5
            },
            "data": {
                "voxel_size": 0.5,
                "volume_size": 128,
                "batch_size": 4
            },
            "alloy": {
                "name": "Ti-6Al-4V",
                "density": 4.43,
                "melting_point": 1604
            }
        }
        
        # Save and load configuration
        model = MicrostructureDiffusion()
        model.save_config(config, str(config_path))
        
        # Load configuration in new model
        model_2 = MicrostructureDiffusion()
        model_2.load_config(str(config_path))
        
        # Configurations should match
        loaded_config = model_2.get_config()
        assert loaded_config["model"]["diffusion_steps"] == 1000
        assert loaded_config["alloy"]["name"] == "Ti-6Al-4V"
    
    @pytest.mark.integration
    def test_error_recovery_pipeline(self, sample_microstructure):
        """Test error recovery in pipeline."""
        model = MicrostructureDiffusion()
        
        # Test recovery from processing errors
        with patch('microdiff_matdesign.imaging.MicroCTProcessor.preprocess') as mock_preprocess:
            mock_preprocess.side_effect = RuntimeError("Processing failed")
            
            # Should gracefully handle processing failure
            with pytest.raises(RuntimeError):
                processor = MicroCTProcessor()
                processor.preprocess(sample_microstructure)
        
        # Test recovery from model errors
        with patch.object(model, '_forward_diffusion') as mock_forward:
            mock_forward.side_effect = RuntimeError("Model failed")
            
            # Should handle model failure gracefully
            with pytest.raises(RuntimeError):
                model.inverse_design(target_microstructure=sample_microstructure)
    
    @pytest.mark.integration
    @pytest.mark.benchmark
    def test_pipeline_performance(self, benchmark, sample_microstructure):
        """Benchmark complete pipeline performance."""
        model = MicrostructureDiffusion()
        processor = MicroCTProcessor()
        
        def run_complete_pipeline():
            processed = processor.preprocess(sample_microstructure)
            return model.inverse_design(
                target_microstructure=processed,
                num_samples=1
            )
        
        result = benchmark(run_complete_pipeline)
        assert result is not None
        
        # Performance should be reasonable (adjust thresholds as needed)
        # This is highly dependent on hardware
        assert benchmark.stats['mean'] < 30.0  # Should complete within 30 seconds