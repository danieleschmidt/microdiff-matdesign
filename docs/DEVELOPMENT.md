# Development Guide

This guide covers development setup, testing, and contribution workflows for MicroDiff-MatDesign.

## Quick Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/microdiff-matdesign.git
cd microdiff-matdesign
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=microdiff_matdesign

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

## Code Quality

```bash
# Format code
black microdiff_matdesign/ tests/

# Check style
flake8 microdiff_matdesign/ tests/

# Type checking
mypy microdiff_matdesign/
```

## Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Release Process

1. Update version in `setup.py` and `__init__.py`
2. Update `CHANGELOG.md`
3. Create release tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`