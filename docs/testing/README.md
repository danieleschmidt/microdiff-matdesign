# Testing Documentation

This directory contains comprehensive testing documentation for MicroDiff-MatDesign.

## Testing Strategy

### Test Pyramid

```
    E2E Tests (CLI, Full Workflows)
         /\
        /  \
   Integration Tests (Pipelines)
      /\    /\
     /  \  /  \
Unit Tests (Components)
```

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Individual component testing
   - Fast execution (<1s per test)
   - High coverage (>90%)
   - No external dependencies

2. **Integration Tests** (`tests/integration/`)
   - Component interaction testing
   - Medium execution time (1-10s per test)
   - Real data processing workflows
   - Limited external dependencies

3. **End-to-End Tests** (`tests/e2e/`)
   - Complete workflow testing
   - Slower execution (10s-5min per test)
   - CLI interface testing
   - Full system validation

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit           # Unit tests (default)
@pytest.mark.integration    # Integration tests
@pytest.mark.e2e           # End-to-end tests
@pytest.mark.slow          # Slow tests (>10s)
@pytest.mark.gpu           # Requires GPU
@pytest.mark.requires_data # Needs test data files
@pytest.mark.benchmark     # Performance benchmarks
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific category
pytest -m unit
pytest -m integration
pytest -m e2e

# Run specific test file
pytest tests/unit/test_core.py

# Run specific test
pytest tests/unit/test_core.py::TestMicrostructureDiffusion::test_initialization
```

### Test Selection

```bash
# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"

# Run only GPU tests
pytest -m gpu

# Run with coverage
pytest --cov=microdiff_matdesign

# Run in parallel
pytest -n auto
```

### Environment Variables

Control test behavior with environment variables:

```bash
# Force CPU-only tests
FORCE_CPU_TESTS=1 pytest

# Skip data-dependent tests
SKIP_DATA_TESTS=1 pytest

# Set test data directory
TEST_DATA_DIR=/path/to/data pytest
```

## Test Configuration

### pytest.ini Configuration

The project uses comprehensive pytest configuration:

- Coverage reporting (HTML, XML, terminal)
- Strict marker enforcement
- Warning filters for ML libraries
- Timeout protection (5min default)
- Parallel execution support

### Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_microstructure`: Synthetic 3D microstructure
- `sample_parameters`: Realistic process parameters
- `mock_model`: Mock ML model for fast testing
- `device`: Appropriate compute device (CPU/GPU)
- `temp_dir`: Temporary directory for test files

## Test Data

### Synthetic Data

Most tests use synthetic data generated on-demand:

```python
def test_with_synthetic_data(sample_microstructure):
    # Use synthetic microstructure
    assert sample_microstructure.shape == (64, 64, 64)
```

### Real Data

Some tests can use real data when available:

```python
@pytest.mark.requires_data
def test_with_real_data(test_data_dir):
    real_file = test_data_dir / "real_microstructure.npy"
    if not real_file.exists():
        pytest.skip("Real data not available")
    # Use real data
```

## Performance Testing

### Benchmarking

Use pytest-benchmark for performance tests:

```python
@pytest.mark.benchmark
def test_inference_speed(benchmark, sample_microstructure):
    model = MicrostructureDiffusion()
    
    result = benchmark(
        model.inverse_design,
        target_microstructure=sample_microstructure
    )
    
    assert benchmark.stats['mean'] < 1.0  # Should complete in <1s
```

### Memory Testing

Monitor memory usage in tests:

```python
def test_memory_usage(sample_microstructure):
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    
    # Run memory-intensive operation
    model = MicrostructureDiffusion()
    result = model.inverse_design(target_microstructure=sample_microstructure)
    
    memory_after = process.memory_info().rss
    memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
    
    assert memory_used < 1000  # Should use <1GB
```

## GPU Testing

### GPU Requirements

GPU tests require:
- CUDA-compatible GPU
- PyTorch with CUDA support
- Sufficient GPU memory (>2GB)

### GPU Test Examples

```python
@pytest.mark.gpu
def test_gpu_acceleration(sample_microstructure):
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    model = MicrostructureDiffusion(device='cuda')
    result = model.inverse_design(target_microstructure=sample_microstructure)
    
    assert result is not None
    assert model.device.type == 'cuda'
```

## Mocking and Patching

### Model Mocking

Mock expensive model operations:

```python
@patch('microdiff_matdesign.core.DiffusionModel')
def test_with_mock_model(mock_model_class):
    mock_model = Mock()
    mock_model.predict.return_value = {"laser_power": 250.0}
    mock_model_class.return_value = mock_model
    
    # Test uses mock instead of real model
    model = MicrostructureDiffusion()
    result = model.inverse_design(sample_microstructure)
```

### External Dependencies

Mock external services and APIs:

```python
@patch('requests.get')
def test_model_download(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"fake model data"
    
    # Test model downloading without network
    model = MicrostructureDiffusion(pretrained=True)
    model.download_pretrained()
```

## Continuous Integration

### Test Stages

CI pipeline runs tests in stages:

1. **Fast Tests**: Unit tests, linting, type checking
2. **Integration Tests**: Component integration, small datasets
3. **GPU Tests**: GPU-accelerated workflows (if GPU available)
4. **Slow Tests**: End-to-end workflows, large datasets

### Test Matrix

Tests run on multiple environments:
- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- PyTorch versions: Latest stable, LTS
- Hardware: CPU-only, GPU-enabled

## Test Maintenance

### Coverage Goals

- **Unit tests**: >90% line coverage
- **Integration tests**: >80% feature coverage
- **E2E tests**: >70% workflow coverage

### Performance Baselines

Maintain performance baselines:
- Inference time: <1s per sample
- Memory usage: <2GB per model
- Training convergence: <100 epochs

### Test Review

Regular test maintenance:
- Remove obsolete tests
- Update test data
- Optimize slow tests
- Add tests for new features

## Troubleshooting

### Common Issues

1. **GPU Tests Failing**
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Run CPU-only tests
   FORCE_CPU_TESTS=1 pytest
   ```

2. **Memory Errors**
   ```bash
   # Reduce batch sizes in tests
   pytest --tb=short -v
   
   # Monitor memory usage
   pytest --memory-profiler
   ```

3. **Slow Tests**
   ```bash
   # Profile slow tests
   pytest --durations=10
   
   # Skip slow tests during development
   pytest -m "not slow"
   ```

4. **Data Dependencies**
   ```bash
   # Check data availability
   ls tests/data/
   
   # Skip data-dependent tests
   pytest -m "not requires_data"
   ```

For more troubleshooting help, see the main project documentation or create an issue on GitHub.