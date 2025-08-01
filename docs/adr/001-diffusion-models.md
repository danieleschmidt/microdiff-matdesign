# ADR-001: Use Diffusion Models for Inverse Design

## Status
Accepted

## Context
Traditional materials design relies on forward simulation (parameters → microstructure → properties), which is computationally expensive and doesn't directly solve the inverse problem of finding optimal process parameters for desired properties.

Recent advances in diffusion models have shown success in image generation and have been adapted for scientific applications. Berkeley's 2025 microstructure diffusion paper demonstrated feasibility for materials applications.

## Decision
Implement diffusion models as the core technology for inverse materials design, specifically:

1. **Denoising Diffusion Probabilistic Models (DDPM)** for parameter generation
2. **3D U-Net architecture** for volumetric microstructure processing
3. **Conditional generation** using microstructure features as conditioning signals
4. **Classifier-free guidance** for controllable parameter generation

## Consequences

### Positive
- Direct solution to inverse design problem
- Handles complex parameter spaces naturally
- Supports uncertainty quantification
- Leverages recent ML advances

### Negative
- Requires large training datasets
- Computationally intensive training
- Model interpretability challenges
- Dependency on GPU resources

### Mitigation
- Provide pre-trained models for common alloys
- Implement efficient inference pipelines
- Add feature importance analysis
- Support CPU-only deployment modes