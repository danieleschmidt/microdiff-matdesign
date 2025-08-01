.PHONY: help clean test install dev-install lint format type-check security-check build docs docker-build docker-test docker-dev
.DEFAULT_GOAL := help

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

help: ## Show this help message
	@echo -e '$(BLUE)MicroDiff-MatDesign Makefile$(NC)'
	@echo -e '$(BLUE)Usage: make [target]$(NC)'
	@echo ''
	@echo -e '$(YELLOW)Development Targets:$(NC)'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: ## Clean build artifacts and cache
	@echo -e '$(YELLOW)Cleaning build artifacts...$(NC)'
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/  
	rm -rf .tox/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo -e '$(GREEN)Clean complete!$(NC)'

test: ## Run tests
	@echo -e '$(YELLOW)Running tests...$(NC)'
	python -m pytest tests/ -v --tb=short

test-fast: ## Run fast tests (unit tests only, no GPU)
	@echo -e '$(YELLOW)Running fast tests...$(NC)'
	python -m pytest tests/unit/ -v -m "not slow and not gpu"

test-gpu: ## Run GPU tests
	@echo -e '$(YELLOW)Running GPU tests...$(NC)'
	python -m pytest tests/ -v -m gpu

test-integration: ## Run integration tests
	@echo -e '$(YELLOW)Running integration tests...$(NC)'
	python -m pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo -e '$(YELLOW)Running end-to-end tests...$(NC)'
	python -m pytest tests/e2e/ -v

test-coverage: ## Run tests with coverage report
	@echo -e '$(YELLOW)Running tests with coverage...$(NC)'
	python -m pytest tests/ --cov=microdiff_matdesign --cov-report=html --cov-report=term-missing

install: ## Install package
	@echo -e '$(YELLOW)Installing package...$(NC)'
	pip install -e .

dev-install: ## Install package in development mode with all dependencies
	@echo -e '$(YELLOW)Installing development dependencies...$(NC)'
	pip install -e ".[dev,gpu,full]"
	pre-commit install
	@echo -e '$(GREEN)Development setup complete!$(NC)'

lint: ## Run linting
	@echo -e '$(YELLOW)Running linting...$(NC)'
	ruff check microdiff_matdesign tests
	bandit -r microdiff_matdesign -f json -o bandit-report.json

format: ## Format code
	@echo -e '$(YELLOW)Formatting code...$(NC)'
	black microdiff_matdesign tests
	isort microdiff_matdesign tests
	ruff --fix microdiff_matdesign tests

type-check: ## Run type checking
	@echo -e '$(YELLOW)Running type checks...$(NC)'
	mypy microdiff_matdesign

security-check: ## Run security checks
	@echo -e '$(YELLOW)Running security checks...$(NC)'
	bandit -r microdiff_matdesign
	safety check --json

quality-check: ## Run all quality checks
	@echo -e '$(YELLOW)Running all quality checks...$(NC)'
	make format
	make lint  
	make type-check
	make security-check
	@echo -e '$(GREEN)All quality checks passed!$(NC)'

build: ## Build package
	@echo -e '$(YELLOW)Building package...$(NC)'
	python -m build
	@echo -e '$(GREEN)Build complete!$(NC)'

docs: ## Build documentation
	@echo -e '$(YELLOW)Building documentation...$(NC)'
	cd docs && sphinx-build -b html . _build/html
	@echo -e '$(GREEN)Documentation built! Open docs/_build/html/index.html$(NC)'

docs-serve: ## Serve documentation locally
	@echo -e '$(YELLOW)Serving documentation at http://localhost:8000$(NC)'
	cd docs/_build/html && python -m http.server 8000

# Docker targets
docker-build: ## Build Docker image
	@echo -e '$(YELLOW)Building Docker image...$(NC)'
	docker build -t microdiff-matdesign:latest .
	@echo -e '$(GREEN)Docker image built!$(NC)'

docker-build-dev: ## Build development Docker image
	@echo -e '$(YELLOW)Building development Docker image...$(NC)'
	docker build --target development -t microdiff-matdesign:dev .

docker-build-prod: ## Build production Docker image
	@echo -e '$(YELLOW)Building production Docker image...$(NC)'
	docker build --target production -t microdiff-matdesign:prod .

docker-test: ## Run tests in Docker
	@echo -e '$(YELLOW)Running tests in Docker...$(NC)'
	docker build --target testing -t microdiff-matdesign:test .
	docker run --rm microdiff-matdesign:test

docker-dev: ## Start development environment with Docker Compose
	@echo -e '$(YELLOW)Starting development environment...$(NC)'
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d microdiff-dev
	@echo -e '$(GREEN)Development environment started!$(NC)'
	@echo -e '$(BLUE)Jupyter Lab: http://localhost:8888$(NC)'
	@echo -e '$(BLUE)TensorBoard: http://localhost:6006$(NC)'

docker-dev-stop: ## Stop development environment
	@echo -e '$(YELLOW)Stopping development environment...$(NC)'
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
	@echo -e '$(GREEN)Development environment stopped!$(NC)'

docker-prod: ## Start production environment
	@echo -e '$(YELLOW)Starting production environment...$(NC)'
	docker-compose up -d microdiff-prod
	@echo -e '$(GREEN)Production environment started!$(NC)'
	@echo -e '$(BLUE)API: http://localhost:8080$(NC)'

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-clean: ## Clean Docker artifacts
	@echo -e '$(YELLOW)Cleaning Docker artifacts...$(NC)'
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo -e '$(GREEN)Docker cleanup complete!$(NC)'

# Benchmark targets
benchmark: ## Run performance benchmarks
	@echo -e '$(YELLOW)Running benchmarks...$(NC)'
	python -m pytest tests/ -m benchmark --benchmark-only

benchmark-save: ## Save benchmark results
	@echo -e '$(YELLOW)Running and saving benchmarks...$(NC)'
	python -m pytest tests/ -m benchmark --benchmark-only --benchmark-json=benchmark_results.json

# Data targets
generate-test-data: ## Generate synthetic test data
	@echo -e '$(YELLOW)Generating test data...$(NC)'
	python -c "
from tests.conftest import generate_synthetic_microstructure, generate_parameter_set
import numpy as np
import os

os.makedirs('tests/data', exist_ok=True)

# Generate sample microstructures
for alloy in ['Ti-6Al-4V', 'Inconel718', 'AlSi10Mg']:
    volume = generate_synthetic_microstructure((64, 64, 64))
    filename = f'tests/data/sample_{alloy.lower().replace(\"-\", \"\").replace(\".\", \"\")}.npy'
    np.save(filename, volume)
    print(f'Generated {filename}')

print('Test data generation complete!')
"
	@echo -e '$(GREEN)Test data generated!$(NC)'

# Release targets
version-patch: ## Bump patch version
	@echo -e '$(YELLOW)Bumping patch version...$(NC)'
	bump2version patch

version-minor: ## Bump minor version
	@echo -e '$(YELLOW)Bumping minor version...$(NC)'
	bump2version minor

version-major: ## Bump major version
	@echo -e '$(YELLOW)Bumping major version...$(NC)'
	bump2version major

# Utility targets
check-deps: ## Check for dependency updates
	@echo -e '$(YELLOW)Checking for dependency updates...$(NC)'
	pip list --outdated

setup-env: ## Set up development environment from scratch
	@echo -e '$(YELLOW)Setting up development environment...$(NC)'
	python -m venv venv
	source venv/bin/activate && make dev-install
	make generate-test-data
	@echo -e '$(GREEN)Development environment setup complete!$(NC)'
	@echo -e '$(BLUE)Activate with: source venv/bin/activate$(NC)'

# CI/CD simulation
ci-test: ## Run CI-style tests locally
	@echo -e '$(YELLOW)Running CI tests...$(NC)'
	make quality-check
	make test-coverage
	make build
	@echo -e '$(GREEN)CI tests passed!$(NC)'

# All-in-one targets
all: clean format lint type-check test build docs ## Run all checks and build
	@echo -e '$(GREEN)All targets completed successfully!$(NC)'

dev-setup: dev-install generate-test-data ## Complete development setup
	@echo -e '$(GREEN)Development setup complete!$(NC)'

# Show system info
info: ## Show system information
	@echo -e '$(BLUE)System Information:$(NC)'
	@echo -e '$(YELLOW)Python:$(NC) $(shell python --version)'
	@echo -e '$(YELLOW)Pip:$(NC) $(shell pip --version)'
	@echo -e '$(YELLOW)Docker:$(NC) $(shell docker --version 2>/dev/null || echo "Not installed")'
	@echo -e '$(YELLOW)CUDA:$(NC) $(shell python -c "import torch; print(f'Available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not installed")'
	@echo -e '$(YELLOW)Git:$(NC) $(shell git --version)'