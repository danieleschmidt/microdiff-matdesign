# Architecture Overview

MicroDiff-MatDesign is structured as a modular Python package for AI-driven materials optimization.

## Core Components

### 1. Core Module (`microdiff_matdesign.core`)
- `MicrostructureDiffusion`: Main diffusion model class
- Handles model initialization, training, and inference
- Supports multiple alloys and manufacturing processes

### 2. Imaging Module (`microdiff_matdesign.imaging`)
- `MicroCTProcessor`: Micro-CT image processing pipeline
- Image preprocessing, denoising, and enhancement
- Phase segmentation and feature extraction

### 3. Models Module (`microdiff_matdesign.models`)
- Diffusion model architectures (U-Net, Transformer)
- Conditioning mechanisms for guided generation
- Uncertainty quantification components

### 4. Process Modules (`microdiff_matdesign.processes`)
- Process-specific implementations (LPBF, EBM, DED)
- Manufacturing constraints and optimization
- Multi-objective optimization algorithms

## Data Flow

```
Micro-CT Image → Preprocessing → Feature Extraction
                                        ↓
Process Parameters ← Diffusion Model ← Microstructure Features
```

## Design Principles

1. **Modularity**: Each component is independently testable
2. **Extensibility**: Easy to add new alloys and processes
3. **Performance**: GPU acceleration where beneficial
4. **Usability**: Simple API for common use cases

## Extension Points

- Add new alloy support via configuration
- Implement custom diffusion architectures
- Extend preprocessing pipelines
- Add manufacturing process modules