# MicroDiff-MatDesign

[![CI/CD](https://github.com/danieleschmidt/microdiff-matdesign/workflows/CI/badge.svg)](https://github.com/danieleschmidt/microdiff-matdesign/actions)
[![Coverage](https://codecov.io/gh/danieleschmidt/microdiff-matdesign/branch/main/graph/badge.svg)](https://codecov.io/gh/danieleschmidt/microdiff-matdesign)
[![Documentation](https://readthedocs.org/projects/microdiff-matdesign/badge/?version=latest)](https://microdiff-matdesign.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/microdiff-matdesign.svg)](https://badge.fury.io/py/microdiff-matdesign)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Diffusion model framework for inverse material design that transforms micro-CT images into printable alloy process parameters. Based on Berkeley's microstructure diffusion paper (April 2025), this tool enables AI-driven optimization of material properties through generative modeling.

> ⚠️ **Enterprise-Ready SDLC Implementation** - This repository includes comprehensive SDLC infrastructure with automated testing, security scanning, monitoring, and deployment pipelines. See [SDLC Documentation](docs/) for details.

## Overview

MicroDiff-MatDesign uses state-of-the-art diffusion models to solve the inverse problem in materials science: given desired microstructural properties, what manufacturing parameters will produce them? The system learns the complex relationship between process parameters and resulting microstructures, enabling rapid material optimization.

## Key Features

- **Inverse Design**: Generate process parameters from target microstructures
- **Multi-Scale Modeling**: Handle features from nanometers to millimeters
- **Process Optimization**: Optimize for multiple objectives (strength, ductility, cost)
- **Uncertainty Quantification**: Confidence bounds on predicted parameters
- **Real-Time Generation**: Sub-second parameter generation after training
- **Multi-Alloy Support**: Pre-trained models for Ti-6Al-4V, Inconel 718, AlSi10Mg

## Installation

```bash
# Basic installation
pip install microdiff-matdesign

# With GPU acceleration
pip install microdiff-matdesign[gpu]

# With all analysis tools
pip install microdiff-matdesign[full]

# From source
git clone https://github.com/danieleschmidt/microdiff-matdesign
cd microdiff-matdesign
pip install -e ".[dev]"
```

## Quick Start

### Basic Inverse Design

```python
from microdiff_matdesign import MicrostructureDiffusion
from microdiff_matdesign.imaging import MicroCTProcessor

# Load and process micro-CT image
processor = MicroCTProcessor()
microstructure = processor.load_image(
    "target_microstructure.tif",
    voxel_size=0.5  # micrometers
)

# Initialize diffusion model
model = MicrostructureDiffusion(
    alloy="Ti-6Al-4V",
    process="laser_powder_bed_fusion",
    pretrained=True
)

# Generate process parameters
parameters = model.inverse_design(
    target_microstructure=microstructure,
    num_samples=10,
    guidance_scale=7.5
)

print("Optimal parameters:")
print(f"Laser power: {parameters.laser_power} W")
print(f"Scan speed: {parameters.scan_speed} mm/s")
print(f"Layer thickness: {parameters.layer_thickness} μm")
print(f"Hatch spacing: {parameters.hatch_spacing} μm")
```

### Training Custom Models

```python
from microdiff_matdesign import train_diffusion_model
from microdiff_matdesign.datasets import MicrostructureDataset

# Prepare dataset
dataset = MicrostructureDataset(
    image_dir="microct_images/",
    parameter_file="process_parameters.csv",
    augment=True,
    normalize=True
)

# Train diffusion model
model = train_diffusion_model(
    dataset=dataset,
    architecture="unet3d",
    diffusion_steps=1000,
    batch_size=8,
    epochs=500,
    learning_rate=1e-4
)

# Validate on test set
metrics = model.validate(dataset.test)
print(f"Parameter prediction MAE: {metrics.mae:.3f}")
print(f"Microstructure similarity: {metrics.ssim:.3f}")
```

## Architecture

```
microdiff-matdesign/
├── microdiff_matdesign/
│   ├── models/
│   │   ├── diffusion/      # Diffusion model architectures
│   │   ├── encoders/       # Microstructure encoders
│   │   ├── decoders/       # Parameter decoders
│   │   └── conditioning/   # Conditional generation
│   ├── imaging/
│   │   ├── preprocessing/  # CT image processing
│   │   ├── segmentation/   # Phase segmentation
│   │   └── features/       # Microstructure descriptors
│   ├── optimization/
│   │   ├── objectives/     # Multi-objective functions
│   │   ├── constraints/    # Manufacturing constraints
│   │   └── solvers/        # Optimization algorithms
│   ├── processes/
│   │   ├── lpbf/          # Laser powder bed fusion
│   │   ├── ebm/           # Electron beam melting
│   │   └── ded/           # Directed energy deposition
│   └── analysis/          # Post-processing tools
├── pretrained/            # Pre-trained models
├── examples/              # Example notebooks
└── benchmarks/            # Performance benchmarks
```

## Microstructure Processing

### Image Preprocessing

```python
from microdiff_matdesign.imaging import MicroCTProcessor

processor = MicroCTProcessor()

# Load and preprocess 3D micro-CT scan
volume = processor.load_volume(
    "scan_directory/",
    file_pattern="slice_*.tif"
)

# Denoise and enhance
processed = processor.preprocess(
    volume,
    denoise_method="bm4d",
    enhance_contrast=True,
    remove_artifacts=True
)

# Segment phases
phases = processor.segment_phases(
    processed,
    num_phases=3,
    method="watershed"
)
```

### Feature Extraction

```python
from microdiff_matdesign.imaging import MicrostructureFeatures

# Extract quantitative features
features = MicrostructureFeatures()

descriptors = features.extract(
    microstructure=phases,
    features=[
        "grain_size_distribution",
        "phase_fractions", 
        "texture_coefficients",
        "porosity",
        "surface_roughness"
    ]
)

# Visualize features
features.plot_grain_size_distribution(descriptors)
features.plot_texture_pole_figures(descriptors)
```

## Diffusion Models

### Model Architectures

```python
from microdiff_matdesign.models import DiffusionUNet3D, DiffusionTransformer

# 3D U-Net for volumetric data
unet_model = DiffusionUNet3D(
    in_channels=1,
    out_channels=8,  # Number of process parameters
    base_channels=64,
    channel_multipliers=[1, 2, 4, 8],
    attention_resolutions=[16, 8],
    num_res_blocks=2
)

# Transformer for irregular data
transformer_model = DiffusionTransformer(
    input_dim=512,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_ratio=4.0
)
```

### Conditional Generation

```python
from microdiff_matdesign.models import ConditionalDiffusion

# Generate with specific constraints
model = ConditionalDiffusion(pretrained="ti64_lpbf")

# Specify desired properties
conditions = {
    "tensile_strength": 1200,  # MPa
    "elongation": 10,          # %
    "density": 0.98,           # relative
    "grain_size": 50           # micrometers
}

# Generate parameters satisfying conditions
parameters = model.conditional_generation(
    conditions=conditions,
    num_proposals=100,
    select_best=True
)
```

### Uncertainty Quantification

```python
from microdiff_matdesign.uncertainty import BayesianDiffusion

# Bayesian diffusion for uncertainty
bayesian_model = BayesianDiffusion(
    base_model=model,
    num_posterior_samples=10
)

# Get parameters with uncertainty
params, uncertainty = bayesian_model.predict_with_uncertainty(
    microstructure,
    confidence_level=0.95
)

print("Parameter ranges (95% CI):")
for param, (low, high) in uncertainty.items():
    print(f"{param}: [{low:.2f}, {high:.2f}]")
```

## Process-Specific Models

### Laser Powder Bed Fusion (LPBF)

```python
from microdiff_matdesign.processes import LPBFModel

lpbf = LPBFModel(
    machine="EOS_M290",
    powder_properties={
        "particle_size_d50": 35,  # micrometers
        "flowability": 25,        # s/50g
        "apparent_density": 2.5   # g/cm³
    }
)

# Optimize for specific microstructure
optimal_params = lpbf.optimize(
    target_microstructure=microstructure,
    constraints={
        "min_density": 0.995,
        "max_surface_roughness": 10,  # Ra in micrometers
        "build_rate": "maximize"
    }
)

## 🔬 Advanced Research Capabilities

**NEW**: Enhanced with comprehensive academic research framework including:

- **📊 Benchmarking Suite**: Rigorous comparative evaluation with statistical significance testing
- **🔄 Reproducibility Controls**: Complete experimental reproducibility with cross-platform validation  
- **📝 Publication Tools**: Automated manuscript generation and research artifact preservation
- **📈 Statistical Analysis**: Effect size computation, confidence intervals, and significance testing
- **🎯 Academic Standards**: Full compliance with peer-review and open science requirements

### Research Features

```python
from microdiff_matdesign.research import (
    BenchmarkSuite, ReproducibilityManager, PublicationManager
)

# Comprehensive model benchmarking
benchmark = BenchmarkSuite()
results = benchmark.run_comprehensive_benchmark(
    config=experiment_config,
    models=['baseline_diffusion', 'physics_informed', 'bayesian_uncertainty']
)

# Reproducibility validation
repro_manager = ReproducibilityManager()
validation = repro_manager.validate_reproducibility(experiment_func, n_runs=5)

# Publication-ready output
pub_manager = PublicationManager(config=publication_config)
manuscript = pub_manager.generate_complete_manuscript(results, validation)
```

## 🏗️ Development & SDLC

This project implements a comprehensive Software Development Life Cycle (SDLC) with enterprise-grade practices:

### Quick Development Setup

# Clone repository
git clone https://github.com/danieleschmidt/microdiff-matdesign
cd microdiff-matdesign

# Setup development environment (recommended)
docker-compose -f docker-compose.dev.yml up -d

# Or install locally
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/

# Run quality checks
make lint
make security-check
```

### Key SDLC Features

- **🧪 Comprehensive Testing**: Unit, integration, and E2E tests with 85%+ coverage target
- **🔒 Security-First**: Multi-layer security scanning (SAST, SCA, container, IaC, DAST)
- **📊 Quality Monitoring**: Automated code quality tracking with trend analysis
- **🚀 CI/CD Pipeline**: Automated testing, building, and deployment with blue-green strategy
- **📈 Observability**: Prometheus metrics, Grafana dashboards, and structured logging
- **🤖 Automation**: Dependency updates, repository maintenance, and metrics collection
- **📚 Documentation**: Architecture decisions, runbooks, and API documentation

### Repository Structure

```
microdiff-matdesign/
├── microdiff_matdesign/     # Core application code
├── tests/                   # Comprehensive test suite
├── docs/                    # Documentation and guides
├── scripts/                 # Automation and utility scripts
├── config/                  # Configuration files
├── .github/                 # GitHub workflows and templates
└── docker-compose*.yml     # Container orchestration
```

### Development Workflow

1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Develop with Quality Gates**: Pre-commit hooks ensure code quality
3. **Test Thoroughly**: Run full test suite before pushing
4. **Create Pull Request**: Automated CI/CD pipeline validates changes
5. **Code Review**: Required reviews from code owners
6. **Deploy**: Automated deployment to staging, then production

### Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)**: System design and components
- **[Contributing Guide](CONTRIBUTING.md)**: Development guidelines and standards
- **[Setup Guide](docs/SETUP_REQUIRED.md)**: Manual setup requirements
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)**: Complete SDLC overview
- **[Monitoring Guide](docs/monitoring/README.md)**: Observability and alerting
- **[Workflow Documentation](docs/workflows/README.md)**: CI/CD pipeline details

### Getting Help

- **Issues**: Use GitHub issues with appropriate labels
- **Security**: Follow [security policy](SECURITY.md) for reporting vulnerabilities
- **Discussions**: Join GitHub Discussions for questions and ideas
- **Contributing**: See [contributing guidelines](CONTRIBUTING.md) for getting started
