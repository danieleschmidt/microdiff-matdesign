# Test Fixtures

This directory contains shared test fixtures and utilities for MicroDiff-MatDesign tests.

## Contents

### Data Fixtures
- `microstructures.py`: Sample microstructure data generators
- `parameters.py`: Sample process parameter generators
- `alloys.py`: Material property fixtures
- `processes.py`: Manufacturing process fixtures

### Mock Fixtures
- `models.py`: Mock model implementations for testing
- `datasets.py`: Mock dataset loaders
- `processors.py`: Mock image processors

### Validation Fixtures
- `golden_data.py`: Golden reference data for validation tests
- `benchmarks.py`: Performance benchmark datasets

## Usage

Import fixtures in your tests:

```python
from tests.fixtures.microstructures import sample_ti64_microstructure
from tests.fixtures.parameters import lpbf_parameter_set
from tests.fixtures.models import mock_diffusion_model

def test_example(sample_ti64_microstructure, lpbf_parameter_set):
    # Use fixtures in your test
    pass
```

## Adding New Fixtures

When adding new fixtures:

1. Place them in the appropriate module based on their type
2. Use descriptive names that indicate the fixture purpose
3. Include docstrings explaining the fixture behavior
4. Add parametrized versions for testing multiple scenarios
5. Consider scope (session, module, function) for performance

## Fixture Scopes

- **session**: Data that doesn't change (reference microstructures, material properties)
- **module**: Test-specific configurations
- **function**: Test instance data that may be modified

## Performance Considerations

Large fixtures (volumetric data, trained models) should use session scope to avoid recreation overhead. For test isolation, create copies when tests need to modify data.