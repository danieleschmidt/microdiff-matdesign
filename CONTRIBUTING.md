# Contributing to MicroDiff-MatDesign

We welcome contributions to MicroDiff-MatDesign! This document provides guidelines for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/microdiff-matdesign.git
   cd microdiff-matdesign
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make changes and test:
   ```bash
   pytest tests/
   black microdiff_matdesign/
   flake8 microdiff_matdesign/
   ```

3. Commit with clear messages:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

4. Push and create a pull request

## Code Standards

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Add type hints where appropriate
- Write docstrings for public functions
- Include tests for new functionality

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

For coverage reports:
```bash
pytest --cov=microdiff_matdesign tests/
```

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)