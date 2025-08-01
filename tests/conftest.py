"""Pytest configuration and shared fixtures for MicroDiff-MatDesign tests."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session") 
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def sample_microstructure() -> np.ndarray:
    """Generate a sample 3D microstructure for testing."""
    # Create a simple 3D volume with different phases
    volume = np.zeros((64, 64, 64), dtype=np.uint8)
    
    # Add some spherical grains
    center = (32, 32, 32)
    radius = 15
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if dist < radius:
                    volume[i, j, k] = 1
                elif dist < radius + 5:
                    volume[i, j, k] = 2
    
    return volume


@pytest.fixture(scope="session")
def sample_parameters() -> Dict[str, float]:
    """Generate sample process parameters for testing."""
    return {
        "laser_power": 280.0,  # W
        "scan_speed": 1200.0,  # mm/s
        "layer_thickness": 30.0,  # μm
        "hatch_spacing": 120.0,  # μm
        "beam_diameter": 80.0,  # μm
        "preheating_temp": 80.0,  # °C
        "chamber_pressure": 1e-5,  # Pa
        "scan_strategy": 0.0,  # categorical: 0=raster, 1=spiral, etc.
    }


@pytest.fixture
def mock_model() -> Mock:
    """Create a mock diffusion model for testing."""
    model = Mock()
    model.predict.return_value = {
        "laser_power": 250.0,
        "scan_speed": 1000.0, 
        "layer_thickness": 25.0,
        "hatch_spacing": 100.0
    }
    model.predict_with_uncertainty.return_value = (
        {"laser_power": 250.0, "scan_speed": 1000.0},
        {"laser_power": (240.0, 260.0), "scan_speed": (950.0, 1050.0)}
    )
    return model


@pytest.fixture
def mock_image_processor() -> Mock:
    """Create a mock image processor for testing."""
    processor = Mock()
    processor.load_volume.return_value = np.random.rand(128, 128, 128)
    processor.preprocess.return_value = np.random.rand(128, 128, 128)
    processor.segment_phases.return_value = np.random.randint(0, 3, (128, 128, 128))
    return processor


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing (CPU or GPU)."""
    if torch.cuda.is_available() and not os.getenv("FORCE_CPU_TESTS"):
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def small_model_config() -> Dict[str, Any]:
    """Configuration for a small model suitable for testing."""
    return {
        "architecture": "unet3d",
        "in_channels": 1,
        "out_channels": 8,
        "base_channels": 16,  # Reduced for testing
        "channel_multipliers": [1, 2],  # Fewer layers
        "attention_resolutions": [],  # No attention for speed
        "num_res_blocks": 1,  # Minimal blocks
        "diffusion_steps": 100,  # Fewer steps for testing
    }


@pytest.fixture
def sample_dataset_config() -> Dict[str, Any]:
    """Configuration for a sample dataset."""
    return {
        "data_dir": "tests/data/microct",
        "batch_size": 2,
        "num_workers": 0,  # No multiprocessing in tests
        "shuffle": False,  # Deterministic for testing
        "pin_memory": False,
        "drop_last": False,
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration for testing."""
    return {
        "model": {
            "architecture": "unet3d",
            "diffusion_steps": 1000,
            "guidance_scale": 7.5,
        },
        "data": {
            "voxel_size": 0.5,
            "volume_size": 128,
            "batch_size": 4,
        },
        "training": {
            "learning_rate": 1e-4,
            "epochs": 100,
            "warmup_steps": 1000,
        },
        "alloy": {
            "name": "Ti-6Al-4V",
            "density": 4.43,
            "melting_point": 1604,
        },
        "process": {
            "type": "laser_powder_bed_fusion",
            "machine": "test_machine",
        }
    }


@pytest.fixture
def capture_logs():
    """Fixture to capture log messages during tests."""
    with patch('logging.getLogger') as mock_logger:
        mock_logger.return_value.info = Mock()
        mock_logger.return_value.warning = Mock() 
        mock_logger.return_value.error = Mock()
        mock_logger.return_value.debug = Mock()
        yield mock_logger.return_value


# Skip tests that require GPU if not available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU hardware"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_slow = pytest.mark.skip(reason="Slow tests disabled")
    
    for item in items:
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
            
        # Skip slow tests if requested
        if "slow" in item.keywords and config.getoption("--no-slow", default=False):
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--no-slow",
        action="store_true",
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--gpu-only",
        action="store_true", 
        default=False,
        help="Run only GPU tests"
    )


# Benchmark configuration
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "min_rounds": 5,
        "max_time": 10.0,
        "warmup": True,
        "warmup_iterations": 2,
    }


# Environment variable fixtures
@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for testing."""
    # Remove any environment variables that might interfere with tests
    env_vars_to_remove = [
        "CUDA_VISIBLE_DEVICES",
        "DEVICE", 
        "NUM_WORKERS",
        "BATCH_SIZE",
        "MODEL_DIR",
        "DATA_DIR"
    ]
    
    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)


# Helper functions for test data generation
def generate_synthetic_microstructure(
    shape: tuple = (64, 64, 64),
    num_phases: int = 3,
    grain_size_range: tuple = (5, 15),
    porosity: float = 0.05
) -> np.ndarray:
    """Generate synthetic microstructure data for testing."""
    volume = np.zeros(shape, dtype=np.uint8)
    
    # Add random grains
    num_grains = np.random.randint(20, 50)
    for _ in range(num_grains):
        center = tuple(np.random.randint(0, s) for s in shape)
        radius = np.random.randint(*grain_size_range)
        phase = np.random.randint(1, num_phases + 1)
        
        # Create spherical grain
        for i in range(max(0, center[0] - radius), min(shape[0], center[0] + radius)):
            for j in range(max(0, center[1] - radius), min(shape[1], center[1] + radius)):
                for k in range(max(0, center[2] - radius), min(shape[2], center[2] + radius)):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    if dist < radius:
                        volume[i, j, k] = phase
    
    # Add porosity
    if porosity > 0:
        num_pores = int(np.prod(shape) * porosity)
        pore_indices = np.random.choice(np.prod(shape), num_pores, replace=False)
        flat_volume = volume.flatten()
        flat_volume[pore_indices] = 0
        volume = flat_volume.reshape(shape)
    
    return volume


def generate_parameter_set(
    alloy: str = "Ti-6Al-4V",
    process: str = "laser_powder_bed_fusion"
) -> Dict[str, float]:
    """Generate realistic parameter sets for testing."""
    if process == "laser_powder_bed_fusion":
        if alloy == "Ti-6Al-4V":
            return {
                "laser_power": np.random.uniform(200, 400),
                "scan_speed": np.random.uniform(800, 1600),
                "layer_thickness": np.random.uniform(20, 50),
                "hatch_spacing": np.random.uniform(80, 160),
                "beam_diameter": np.random.uniform(60, 120),
                "preheating_temp": np.random.uniform(20, 200),
            }
    
    # Default fallback
    return {
        "param_1": np.random.uniform(0, 1),
        "param_2": np.random.uniform(0, 1),
        "param_3": np.random.uniform(0, 1),
        "param_4": np.random.uniform(0, 1),
    }