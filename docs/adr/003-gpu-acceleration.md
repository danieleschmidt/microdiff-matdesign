# ADR-003: GPU Acceleration Strategy

## Status
Accepted

## Context
Diffusion models are computationally intensive, especially for 3D volumetric data. Training and inference can benefit significantly from GPU acceleration, but we need to support users without GPU access.

## Decision
Implement hybrid CPU/GPU strategy:

1. **PyTorch backend** with automatic device detection
2. **Optional GPU dependencies** via `[gpu]` extra
3. **Graceful fallbacks** to CPU implementations
4. **Memory management** for large 3D volumes
5. **Batch processing** for multiple samples

### GPU Optimization
- Use mixed precision training (FP16)
- Implement gradient checkpointing for memory efficiency
- Support multi-GPU training with DDP
- Optimize data loading pipelines

### CPU Fallbacks
- Reduced model sizes for CPU inference
- Progressive processing for large volumes
- NumPy implementations for core algorithms
- Warning messages for performance expectations

## Consequences

### Positive
- Accessible to users without GPUs
- Optimal performance with GPU hardware
- Scalable to large datasets
- Future-proof for new accelerators

### Negative
- Complex dependency management
- Testing overhead (CPU + GPU variants)
- Memory management complexity
- Performance tuning required

### Mitigation
- Comprehensive testing on both backends
- Clear documentation of hardware requirements
- Performance benchmarks and recommendations
- Container images with optimized environments