.PHONY: help install install-dev test lint format type-check security clean build docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=microdiff_matdesign --cov-report=term-missing

test-fast: ## Run tests without coverage
	pytest tests/ -v -x

lint: ## Run linting
	flake8 microdiff_matdesign tests
	black --check microdiff_matdesign tests

format: ## Format code
	black microdiff_matdesign tests
	isort microdiff_matdesign tests

type-check: ## Run type checking
	mypy microdiff_matdesign

security: ## Run security checks
	bandit -r microdiff_matdesign
	safety check

quality: lint type-check security ## Run all quality checks

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

ci: install-dev quality test ## Run full CI pipeline