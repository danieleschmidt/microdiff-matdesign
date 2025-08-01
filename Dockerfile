# Multi-stage Docker build for MicroDiff-MatDesign
# Base image with CUDA support for GPU acceleration
ARG CUDA_VERSION=11.8
ARG PYTHON_VERSION=3.11

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfftw3-dev \
    libtiff5-dev \
    libjpeg-dev \
    libpng-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Development stage with all dependencies
FROM base as development

# Install development and testing dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install -e ".[dev,gpu,full]"

# Install additional development tools
RUN pip install \
    jupyterlab \
    tensorboard \
    wandb \
    mlflow

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev,gpu,full]"

# Expose ports for development services
EXPOSE 8888 6006 3000

# Default command for development
CMD ["bash"]

# Production build stage
FROM base as builder

# Copy requirements and install production dependencies
COPY requirements.txt pyproject.toml setup.py ./
COPY microdiff_matdesign/ ./microdiff_matdesign/

# Install production dependencies only
RUN pip install --no-deps -e ".[gpu]"

# Install additional runtime dependencies
RUN pip install \
    gunicorn \
    uvicorn[standard] \
    prometheus-client

# Production runtime stage
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 as production

ENV DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libtiff5 \
    libjpeg8 \
    libpng16-16 \
    libhdf5-103 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Create non-root user for security
RUN groupadd -r microdiff && useradd -r -g microdiff -u 1000 microdiff

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python*/site-packages/ /usr/local/lib/python*/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --chown=microdiff:microdiff microdiff_matdesign/ ./microdiff_matdesign/
COPY --chown=microdiff:microdiff pyproject.toml setup.py ./

# Install the package
RUN pip install --no-deps -e .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/temp && \
    chown -R microdiff:microdiff /app

# Switch to non-root user
USER microdiff

# Set environment variables for production
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import microdiff_matdesign; print('OK')" || exit 1

# Default command for production
CMD ["python", "-m", "microdiff_matdesign.cli"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Install test data
RUN mkdir -p tests/data && \
    python -c "
import numpy as np
from tests.conftest import generate_synthetic_microstructure, generate_parameter_set

# Generate test microstructures
for alloy in ['Ti-6Al-4V', 'Inconel718', 'AlSi10Mg']:
    volume = generate_synthetic_microstructure((64, 64, 64))
    np.save(f'tests/data/sample_{alloy.lower().replace(\"-\", \"\").replace(\".\", \"\")}.npy', volume)

print('Test data generated')
"

# Run tests to validate build
RUN python -m pytest tests/unit/ -v --tb=short -m "not slow and not gpu" || true

# Benchmarking stage
FROM production as benchmark

# Install benchmarking tools
USER root
RUN pip install pytest-benchmark memory-profiler

# Copy benchmark tests
COPY --chown=microdiff:microdiff tests/benchmarks/ ./tests/benchmarks/

USER microdiff

# Run benchmarks
RUN python -m pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark_results.json || true

# Documentation stage
FROM base as docs

# Install documentation dependencies
RUN pip install \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    nbsphinx \
    sphinx-autodoc-typehints

# Copy source and docs
COPY microdiff_matdesign/ ./microdiff_matdesign/
COPY docs/ ./docs/
COPY pyproject.toml setup.py ./

# Install package for documentation
RUN pip install -e .

# Build documentation
WORKDIR /app/docs
RUN sphinx-build -b html . _build/html

# Serve documentation
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "_build/html"]

# Multi-architecture support
FROM production as multiarch

# Add support for ARM64 and AMD64
ARG TARGETPLATFORM
ARG BUILDPLATFORM

RUN echo "Building for ${TARGETPLATFORM} on ${BUILDPLATFORM}"

# Platform-specific optimizations
RUN if [ "${TARGETPLATFORM}" = "linux/arm64" ]; then \
        echo "Configuring for ARM64..."; \
        export CFLAGS="-O3 -mcpu=native"; \
    elif [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        echo "Configuring for AMD64..."; \
        export CFLAGS="-O3 -march=native"; \
    fi

# Default production image
FROM production