# ADR-002: Python Package Structure

## Status
Accepted

## Context
Need to organize code for maintainability, extensibility, and clear separation of concerns. The package serves multiple user types: researchers, engineers, and software developers.

## Decision
Implement modular package structure with clear domain separation:

```
microdiff_matdesign/
├── core.py              # Main API and orchestration
├── models/              # Diffusion model architectures
├── imaging/             # Image processing and feature extraction
├── processes/           # Manufacturing process modules
├── optimization/        # Multi-objective optimization
├── datasets/           # Data loading and preprocessing
├── utils/              # Shared utilities
└── cli.py              # Command-line interface
```

### Design Principles
1. **Domain-driven design**: Each module maps to a specific domain
2. **Minimal coupling**: Interfaces between modules are well-defined
3. **Plugin architecture**: New processes and models can be added easily
4. **User-focused API**: Simple interface for common tasks

## Consequences

### Positive
- Clear separation of concerns
- Easy to extend with new processes/alloys
- Testable components
- Multiple user interfaces (API, CLI, notebooks)

### Negative
- Initial complexity for simple use cases
- Risk of over-abstraction
- Import overhead for small tasks

### Mitigation
- Provide high-level convenience functions in `core.py`
- Clear documentation with examples
- Lazy loading of heavy components