"""End-to-end tests for CLI interface."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from click.testing import CliRunner

from microdiff_matdesign.cli import main, inverse_design, train, validate


class TestCLIInterface:
    """End-to-end tests for command line interface."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.e2e
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'inverse-design' in result.output
        assert 'train' in result.output
        assert 'validate' in result.output
    
    @pytest.mark.e2e
    def test_inverse_design_command_help(self):
        """Test inverse design command help."""
        result = self.runner.invoke(main, ['inverse-design', '--help'])
        assert result.exit_code == 0
        assert 'Generate process parameters' in result.output
        assert '--input' in result.output
        assert '--output' in result.output
        assert '--alloy' in result.output
        assert '--process' in result.output
    
    @pytest.mark.e2e
    def test_train_command_help(self):
        """Test train command help."""
        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        assert 'Train a diffusion model' in result.output
        assert '--data-dir' in result.output
        assert '--config' in result.output
        assert '--epochs' in result.output
    
    @pytest.mark.e2e
    def test_validate_command_help(self):
        """Test validate command help."""
        result = self.runner.invoke(main, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'Validate model performance' in result.output
        assert '--model' in result.output
        assert '--test-data' in result.output
    
    @pytest.mark.e2e
    def test_inverse_design_with_numpy_input(self, sample_microstructure):
        """Test inverse design with numpy array input."""
        # Save sample microstructure
        input_file = self.temp_dir / "input.npy"
        output_file = self.temp_dir / "output.json"
        
        import numpy as np
        np.save(input_file, sample_microstructure)
        
        # Run inverse design
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', str(input_file),
            '--output', str(output_file),
            '--alloy', 'Ti-6Al-4V',
            '--process', 'laser_powder_bed_fusion',
            '--num-samples', '3'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Validate output
        with open(output_file) as f:
            output_data = json.load(f)
        
        assert 'parameters' in output_data
        assert 'laser_power' in output_data['parameters']
        assert 'scan_speed' in output_data['parameters']
        assert 'metadata' in output_data
        assert output_data['metadata']['alloy'] == 'Ti-6Al-4V'
    
    @pytest.mark.e2e 
    def test_inverse_design_with_image_directory(self):
        """Test inverse design with image directory input."""
        # Create mock image directory
        image_dir = self.temp_dir / "images"
        image_dir.mkdir()
        
        # Create mock TIFF files
        import numpy as np
        from PIL import Image
        
        for i in range(10):
            slice_data = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            image = Image.fromarray(slice_data)
            image.save(image_dir / f"slice_{i:03d}.tif")
        
        output_file = self.temp_dir / "output.yaml"
        
        # Run inverse design
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', str(image_dir),
            '--output', str(output_file),
            '--format', 'yaml',
            '--alloy', 'Inconel718'
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Validate YAML output
        with open(output_file) as f:
            output_data = yaml.safe_load(f)
        
        assert 'parameters' in output_data
        assert output_data['metadata']['alloy'] == 'Inconel718'
    
    @pytest.mark.e2e
    def test_batch_processing(self, sample_microstructure):
        """Test batch processing multiple files."""
        # Create multiple input files
        input_dir = self.temp_dir / "inputs"
        input_dir.mkdir()
        output_dir = self.temp_dir / "outputs"
        output_dir.mkdir()
        
        import numpy as np
        for i in range(3):
            input_file = input_dir / f"sample_{i}.npy"
            # Add some variation to each sample
            varied_sample = sample_microstructure + np.random.normal(0, 0.1, sample_microstructure.shape)
            np.save(input_file, varied_sample)
        
        # Run batch processing
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', str(input_dir),
            '--output', str(output_dir),
            '--batch',
            '--alloy', 'AlSi10Mg'
        ])
        
        assert result.exit_code == 0
        
        # Check that output files were created
        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) == 3
        
        # Validate each output
        for output_file in output_files:
            with open(output_file) as f:
                data = json.load(f)
            assert 'parameters' in data
            assert data['metadata']['alloy'] == 'AlSi10Mg'
    
    @pytest.mark.e2e
    def test_configuration_file(self):
        """Test using configuration file."""
        # Create configuration file
        config_file = self.temp_dir / "config.yaml"
        config = {
            'model': {
                'architecture': 'unet3d',
                'diffusion_steps': 500,
                'guidance_scale': 5.0
            },
            'alloy': {
                'name': 'Ti-6Al-4V',
                'density': 4.43
            },
            'process': {
                'type': 'laser_powder_bed_fusion',
                'constraints': {
                    'min_density': 0.98,
                    'max_surface_roughness': 15
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Test that config is loaded
        result = self.runner.invoke(main, [
            '--config', str(config_file),
            'inverse-design', '--help'
        ])
        
        assert result.exit_code == 0
    
    @pytest.mark.e2e
    def test_uncertainty_quantification_cli(self, sample_microstructure):
        """Test uncertainty quantification via CLI."""
        input_file = self.temp_dir / "input.npy"
        output_file = self.temp_dir / "output.json"
        
        import numpy as np
        np.save(input_file, sample_microstructure)
        
        # Run with uncertainty quantification
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', str(input_file),
            '--output', str(output_file),
            '--uncertainty',
            '--confidence-level', '0.95',
            '--num-samples', '10'
        ])
        
        assert result.exit_code == 0
        
        # Validate uncertainty output
        with open(output_file) as f:
            output_data = json.load(f)
        
        assert 'parameters' in output_data
        assert 'uncertainty' in output_data
        
        for param in output_data['parameters']:
            assert param in output_data['uncertainty']
            assert 'lower_bound' in output_data['uncertainty'][param]
            assert 'upper_bound' in output_data['uncertainty'][param]
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_training_command(self):
        """Test model training command."""
        # Create mock training data
        data_dir = self.temp_dir / "training_data"
        data_dir.mkdir()
        
        images_dir = data_dir / "images"
        images_dir.mkdir()
        
        # Create mock training images and parameters
        import numpy as np
        import pandas as pd
        
        parameters_data = []
        for i in range(5):  # Small dataset for testing
            # Create mock microstructure
            volume = np.random.randint(0, 3, (32, 32, 32), dtype=np.uint8)
            np.save(images_dir / f"sample_{i}.npy", volume)
            
            # Create corresponding parameters
            parameters_data.append({
                'filename': f"sample_{i}.npy",
                'laser_power': 200 + np.random.normal(0, 50),
                'scan_speed': 1000 + np.random.normal(0, 200),
                'layer_thickness': 30 + np.random.normal(0, 5),
                'hatch_spacing': 120 + np.random.normal(0, 20)
            })
        
        # Save parameters CSV
        parameters_df = pd.DataFrame(parameters_data)
        parameters_df.to_csv(data_dir / "parameters.csv", index=False)
        
        # Create training config
        config_file = self.temp_dir / "train_config.yaml"
        train_config = {
            'data': {
                'image_dir': str(images_dir),
                'parameters_file': str(data_dir / "parameters.csv"),
                'batch_size': 2,
                'validation_split': 0.2
            },
            'model': {
                'architecture': 'unet3d',
                'base_channels': 16  # Small for testing
            },
            'training': {
                'epochs': 2,
                'learning_rate': 1e-4,
                'warmup_steps': 10
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(train_config, f)
        
        # Run training
        result = self.runner.invoke(main, [
            'train',
            '--config', str(config_file),
            '--output-dir', str(self.temp_dir / "model_output")
        ])
        
        assert result.exit_code == 0
        
        # Check that model was saved
        model_dir = self.temp_dir / "model_output"
        assert model_dir.exists()
        assert (model_dir / "model.pth").exists() or any(model_dir.glob("*.pth"))
    
    @pytest.mark.e2e
    def test_validation_command(self, sample_microstructure):
        """Test model validation command."""
        # Create test data
        test_dir = self.temp_dir / "test_data"
        test_dir.mkdir()
        
        images_dir = test_dir / "images"
        images_dir.mkdir()
        
        import numpy as np
        import pandas as pd
        
        # Create test samples
        test_data = []
        for i in range(3):
            volume = sample_microstructure + np.random.normal(0, 0.1, sample_microstructure.shape)
            np.save(images_dir / f"test_{i}.npy", volume)
            
            test_data.append({
                'filename': f"test_{i}.npy",
                'laser_power': 250 + np.random.normal(0, 25),
                'scan_speed': 1200 + np.random.normal(0, 120),
                'layer_thickness': 30 + np.random.normal(0, 3),
                'hatch_spacing': 120 + np.random.normal(0, 12)
            })
        
        pd.DataFrame(test_data).to_csv(test_dir / "test_parameters.csv", index=False)
        
        # Run validation (with pretrained model)
        result = self.runner.invoke(main, [
            'validate',
            '--test-data', str(test_dir),
            '--model', 'pretrained:Ti-6Al-4V',
            '--output', str(self.temp_dir / "validation_results.json")
        ])
        
        # Should complete without error (may have warnings about pretrained model)
        assert result.exit_code == 0
        
        # Check validation results
        results_file = self.temp_dir / "validation_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            assert 'metrics' in results
    
    @pytest.mark.e2e
    def test_verbose_and_quiet_modes(self, sample_microstructure):
        """Test verbose and quiet output modes."""
        input_file = self.temp_dir / "input.npy"
        output_file = self.temp_dir / "output.json"
        
        import numpy as np
        np.save(input_file, sample_microstructure)
        
        # Test verbose mode
        result_verbose = self.runner.invoke(main, [
            '--verbose',
            'inverse-design',
            '--input', str(input_file),
            '--output', str(output_file)
        ])
        
        assert result_verbose.exit_code == 0
        # Verbose mode should produce more output
        verbose_output_lines = len(result_verbose.output.split('\n'))
        
        # Test quiet mode
        result_quiet = self.runner.invoke(main, [
            '--quiet',
            'inverse-design',
            '--input', str(input_file),
            '--output', str(output_file)
        ])
        
        assert result_quiet.exit_code == 0
        # Quiet mode should produce less output
        quiet_output_lines = len(result_quiet.output.split('\n'))
        
        assert verbose_output_lines > quiet_output_lines
    
    @pytest.mark.e2e
    def test_error_handling(self):
        """Test CLI error handling."""
        # Test with non-existent input file
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', 'nonexistent.npy',
            '--output', 'output.json'
        ])
        
        assert result.exit_code != 0
        assert 'Error' in result.output or 'error' in result.output.lower()
        
        # Test with invalid alloy
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', 'dummy.npy',
            '--output', 'output.json',
            '--alloy', 'InvalidAlloy'
        ])
        
        assert result.exit_code != 0
        
        # Test with invalid parameters
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', 'dummy.npy',
            '--output', 'output.json',
            '--num-samples', '-1'
        ])
        
        assert result.exit_code != 0
    
    @pytest.mark.e2e
    def test_output_formats(self, sample_microstructure):
        """Test different output formats."""
        input_file = self.temp_dir / "input.npy"
        import numpy as np
        np.save(input_file, sample_microstructure)
        
        # Test JSON output
        json_output = self.temp_dir / "output.json"
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', str(input_file),
            '--output', str(json_output),
            '--format', 'json'
        ])
        assert result.exit_code == 0
        assert json_output.exists()
        
        # Test YAML output
        yaml_output = self.temp_dir / "output.yaml"
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', str(input_file),
            '--output', str(yaml_output),
            '--format', 'yaml'
        ])
        assert result.exit_code == 0
        assert yaml_output.exists()
        
        # Test CSV output
        csv_output = self.temp_dir / "output.csv"
        result = self.runner.invoke(main, [
            'inverse-design',
            '--input', str(input_file),
            '--output', str(csv_output),
            '--format', 'csv'
        ])
        assert result.exit_code == 0
        assert csv_output.exists()
    
    @pytest.mark.e2e
    @pytest.mark.benchmark
    def test_cli_performance(self, benchmark, sample_microstructure):
        """Benchmark CLI performance."""
        input_file = self.temp_dir / "input.npy"
        output_file = self.temp_dir / "output.json"
        
        import numpy as np
        np.save(input_file, sample_microstructure)
        
        def run_cli():
            result = self.runner.invoke(main, [
                'inverse-design',
                '--input', str(input_file),
                '--output', str(output_file),
                '--num-samples', '1'
            ])
            return result.exit_code
        
        exit_code = benchmark(run_cli)
        assert exit_code == 0