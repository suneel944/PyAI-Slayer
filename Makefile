# PyAI-Slayer Makefile
# Build orchestration and developer commands

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Variables
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
MIN_PYTHON_VERSION := 3.11
PROJECT_NAME := pyai-slayer

# Load .env file if it exists
# Read .env file and set variables (handles comments and empty lines)
ifneq (,$(wildcard .env))
    $(foreach line,$(shell grep -v '^#' .env | grep -v '^$$'),$(eval $(line)))
endif

# Default HEADLESS to true if not set in .env (for CI/CD safety)
# Only set if HEADLESS was not already loaded from .env file
ifndef HEADLESS
    HEADLESS := true
endif
export HEADLESS

# Default PARALLEL_WORKERS to 4 if not set in .env
# Only set if PARALLEL_WORKERS was not already loaded from .env file
ifndef PARALLEL_WORKERS
    PARALLEL_WORKERS := 4
endif
export PARALLEL_WORKERS

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
endif

.PHONY: help check-python venv setup install install-dev clean clean-build clean-cache clean-test clean-venv clean-all test test-unit test-all test-integration test-cov lint format type-check check build ci activate playwright-install

# Default target
help:
	@echo "$(CYAN)PyAI-Slayer - The Open Source AI Testing Arsenal$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(YELLOW)make setup$(NC)              - Complete project setup (check Python, create venv, install deps)"
	@echo "  $(YELLOW)make install$(NC)            - Install package in production mode"
	@echo "  $(YELLOW)make install-dev$(NC)       - Install package with dev dependencies"
	@echo "  $(YELLOW)make test$(NC)               - Run e2e application tests (AI, security, UI - excludes unit tests)"
	@echo "  $(YELLOW)make test-unit$(NC)         - Run unit tests only"
	@echo "  $(YELLOW)make test-all$(NC)          - Run all tests (unit + e2e)"
	@echo "  $(YELLOW)make test-cov$(NC)          - Run all tests with coverage report"
	@echo "  $(YELLOW)make lint$(NC)               - Run linter (ruff)"
	@echo "  $(YELLOW)make format$(NC)            - Format code (ruff format only)"
	@echo "  $(YELLOW)make pre-commit$(NC)        - Run all pre-commit checks (format + lint + type-check + security)"
	@echo "  $(YELLOW)make type-check$(NC)        - Run type checker (mypy)"
	@echo "  $(YELLOW)make check$(NC)             - Run all checks (lint + type-check)"
	@echo "  $(YELLOW)make build$(NC)             - Build distribution packages"
	@echo "  $(YELLOW)make ci$(NC)                - Run CI pipeline (install-dev + lint + type-check + test-unit)"
	@echo "  $(YELLOW)make clean$(NC)             - Remove venv, build artifacts, caches"
	@echo "  $(YELLOW)make playwright-install$(NC) - Install Playwright browsers"
	@echo "  $(YELLOW)make activate$(NC)          - Show venv activation instructions"
	@echo "  $(YELLOW)make metrics-summary$(NC)   - View test metrics summary (requires metrics enabled)"
	@echo "  $(YELLOW)make metrics-export$(NC)     - Export metrics to JSON file"
	@echo ""

# Check Python version
check-python:
	@echo "$(CYAN)Checking Python version...$(NC)"
	@$(PYTHON) --version
	@$(PYTHON) -c "import sys; \
		min_version = tuple(map(int, '$(MIN_PYTHON_VERSION)'.split('.'))); \
		current_version = sys.version_info[:2]; \
		exit(0 if current_version >= min_version else 1)" || \
		(echo "$(RED)Error: Python $(MIN_PYTHON_VERSION)+ required$(NC)" && exit 1)
	@echo "$(GREEN)✓ Python version OK$(NC)"

# Create virtual environment
venv: check-python
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(CYAN)Creating virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)Virtual environment already exists$(NC)"; \
	fi

# Install package in production mode
install: venv
	@echo "$(CYAN)Installing package in production mode...$(NC)"
	@$(VENV_BIN)/pip install --upgrade pip setuptools wheel
	@$(VENV_BIN)/pip install -e .
	@echo "$(GREEN)✓ Package installed$(NC)"

# Install package with dev dependencies
install-dev: venv
	@echo "$(CYAN)Installing package with dev dependencies...$(NC)"
	@$(VENV_BIN)/pip install --upgrade pip setuptools wheel build twine
	@$(VENV_BIN)/pip install -e ".[dev]"
	@echo "$(GREEN)✓ Package installed with dev dependencies$(NC)"

# Complete setup (main entry point)
setup: check-python venv install-dev playwright-install
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Setup complete! ✓$(NC)"
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(CYAN)Next steps:$(NC)"
	@echo "  1. Activate virtual environment:"
	@echo "     $(YELLOW)source $(VENV)/bin/activate$(NC)"
	@echo "  2. Copy and configure environment:"
	@echo "     $(YELLOW)cp .env.example .env$(NC)"
	@echo "     $(YELLOW)# Edit .env with your credentials$(NC)"
	@echo "  3. Run AI chatbot tests:"
	@echo "     $(YELLOW)make test$(NC)"
	@echo "  4. Run all checks:"
	@echo "     $(YELLOW)make check$(NC)"
	@echo ""

# Install Playwright browsers
playwright-install: venv
	@echo "$(CYAN)Installing Playwright browsers...$(NC)"
	@$(VENV_BIN)/playwright install
	@echo "$(GREEN)✓ Playwright browsers installed$(NC)"

# Run e2e application tests (AI, security, UI - excludes unit tests)
test: venv
	@echo "$(CYAN)Running e2e application tests in parallel ($(PARALLEL_WORKERS) workers) with verbose logs...$(NC)"
	@echo "$(YELLOW)HEADLESS=$(HEADLESS)$(NC)"
	@echo "$(YELLOW)PARALLEL_WORKERS=$(PARALLEL_WORKERS)$(NC)"
	@$(VENV_BIN)/pytest tests/e2e/ai/ tests/e2e/security/ tests/e2e/ui/ -n $(PARALLEL_WORKERS) -vv -s --tb=short --log-cli-level=INFO --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"

# Run tests with specific markers
test-integration: venv
	@echo "$(CYAN)Running integration tests in parallel ($(PARALLEL_WORKERS) workers) with verbose logs...$(NC)"
	@echo "$(YELLOW)HEADLESS=$(HEADLESS)$(NC)"
	@echo "$(YELLOW)PARALLEL_WORKERS=$(PARALLEL_WORKERS)$(NC)"
	@$(VENV_BIN)/pytest tests/ -n $(PARALLEL_WORKERS) -vv -s -m "integration" --tb=short --log-cli-level=INFO --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"

# Run all tests (unit + e2e)
test-all: venv
	@echo "$(CYAN)Running all tests (unit + e2e) in parallel ($(PARALLEL_WORKERS) workers) with verbose logs...$(NC)"
	@echo "$(YELLOW)HEADLESS=$(HEADLESS)$(NC)"
	@echo "$(YELLOW)PARALLEL_WORKERS=$(PARALLEL_WORKERS)$(NC)"
	@$(VENV_BIN)/pytest tests/ -n $(PARALLEL_WORKERS) -vv -s --tb=short --log-cli-level=INFO --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"

# Run tests with coverage
test-cov: venv
	@echo "$(CYAN)Running all tests in parallel ($(PARALLEL_WORKERS) workers) with coverage and verbose logs...$(NC)"
	@echo "$(YELLOW)HEADLESS=$(HEADLESS)$(NC)"
	@echo "$(YELLOW)PARALLEL_WORKERS=$(PARALLEL_WORKERS)$(NC)"
	@$(VENV_BIN)/pytest tests/ -n $(PARALLEL_WORKERS) --cov=src \
		--cov-report=html --cov-report=term-missing -vv -s --log-cli-level=INFO --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

# Run linter
lint: venv
	@echo "$(CYAN)Running linter (ruff)...$(NC)"
	@$(VENV_BIN)/ruff check src/ tests/ scripts/
	@echo "$(GREEN)✓ Linting complete$(NC)"

# Format code (uses ruff format only - linting is handled by pre-commit)
format: venv
	@echo "$(CYAN)Formatting code with ruff...$(NC)"
	@$(VENV_BIN)/ruff format src/ tests/ scripts/
	@echo "$(GREEN)✓ Code formatted$(NC)"
	@echo "$(YELLOW)Note: Run 'make pre-commit' for comprehensive checks (formatting + linting + type-check + security)$(NC)"

# Check code formatting (without fixing) - for CI
format-check: venv
	@echo "$(CYAN)Checking code formatting...$(NC)"
	@$(VENV_BIN)/ruff format --check src/ tests/ scripts/
	@echo "$(GREEN)✓ Formatting check complete$(NC)"

# Run type checker
type-check: venv
	@echo "$(CYAN)Running type checker (mypy)...$(NC)"
	@$(VENV_BIN)/mypy src/ scripts/ --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# Run security scan
security-scan: venv
	@echo "$(CYAN)Running security scan (bandit)...$(NC)"
	@$(VENV_BIN)/bandit -r src/ -f json -o bandit-report.json || true
	@echo "$(GREEN)✓ Security scan complete$(NC)"

# Run all checks (linting and type checking only, no tests)
check: lint type-check
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  All checks passed! ✓$(NC)"
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"

# Build distribution packages
build: venv clean-build
	@echo "$(CYAN)Building distribution packages...$(NC)"
	@$(VENV_BIN)/python -m build
	@echo "$(GREEN)✓ Build complete. Packages in dist/$(NC)"

# CI pipeline
ci: install-dev lint type-check test-unit
	@echo ""
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  CI pipeline passed! ✓$(NC)"
	@echo "$(GREEN)════════════════════════════════════════════════════════$(NC)"

# Clean build artifacts
clean-build:
	@echo "$(CYAN)Cleaning build artifacts...$(NC)"
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf docs/_build/
	@rm -rf .tox/
	@echo "$(GREEN)✓ Build artifacts cleaned$(NC)"

# Clean Python caches
clean-cache:
	@echo "$(CYAN)Cleaning Python caches...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf .hypothesis/
	@echo "$(GREEN)✓ Python caches cleaned$(NC)"

# Clean test artifacts
clean-test:
	@echo "$(CYAN)Cleaning test artifacts...$(NC)"
	@rm -rf .coverage
	@rm -rf htmlcov/
	@rm -rf reports/
	@rm -rf screenshots/
	@rm -rf traces/
	@rm -f bandit-report.json
	@rm -f dashboard_data.db
	@echo "$(GREEN)✓ Test artifacts cleaned$(NC)"

# Clean virtual environment
clean-venv:
	@echo "$(CYAN)Removing virtual environment...$(NC)"
	@rm -rf $(VENV)
	@rm -rf .venv
	@rm -rf venv/
	@echo "$(GREEN)✓ Virtual environment removed$(NC)"

# Clean everything
clean: clean-build clean-cache clean-test
	@echo "$(GREEN)✓ Clean complete (venv preserved, use 'make clean-venv' to remove)$(NC)"

# Clean everything including venv
clean-all: clean-build clean-cache clean-test clean-venv
	@echo "$(GREEN)✓ Complete clean finished$(NC)"

# Show activation instructions
activate:
	@echo "$(CYAN)Virtual Environment Activation:$(NC)"
	@echo ""
	@echo "$(GREEN)Linux/macOS:$(NC)"
	@echo "  $(YELLOW)source $(VENV)/bin/activate$(NC)"
	@echo ""
	@echo "$(GREEN)Windows (PowerShell):$(NC)"
	@echo "  $(YELLOW)$(VENV)\\Scripts\\Activate.ps1$(NC)"
	@echo ""
	@echo "$(GREEN)Windows (CMD):$(NC)"
	@echo "  $(YELLOW)$(VENV)\\Scripts\\activate.bat$(NC)"
	@echo ""
	@echo "$(CYAN)To deactivate:$(NC)"
	@echo "  $(YELLOW)deactivate$(NC)"

# Build documentation
docs: venv
	@echo "$(CYAN)Building Sphinx documentation...$(NC)"
	@$(VENV_BIN)/pip install -e ".[dev]" > /dev/null 2>&1 || true
	@cd docs && $(VENV_BIN)/sphinx-build -b html . _build/html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html/$(NC)"

# Serve documentation
docs-serve: docs
	@echo "$(CYAN)Serving documentation at http://localhost:8000$(NC)"
	@cd docs/_build/html && $(VENV_BIN)/python -m http.server 8000

# Run unit tests only
test-unit: venv
	@echo "$(CYAN)Running unit tests in parallel ($(PARALLEL_WORKERS) workers) with verbose logs...$(NC)"
	@echo "$(YELLOW)PARALLEL_WORKERS=$(PARALLEL_WORKERS)$(NC)"
	@$(VENV_BIN)/pytest tests/unit/ -n $(PARALLEL_WORKERS) -vv -s --tb=short --log-cli-level=INFO --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"

# Run unit tests with coverage (for CI)
test-unit-cov: venv
	@echo "$(CYAN)Running unit tests with coverage...$(NC)"
	@$(VENV_BIN)/pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=term-missing

# Run property-based tests
test-property: venv
	@echo "$(CYAN)Running property-based tests...$(NC)"
	@$(VENV_BIN)/pytest tests/unit/test_*_property.py -v

# Install pre-commit hooks
install-hooks: venv
	@echo "$(CYAN)Installing pre-commit hooks...$(NC)"
	@$(VENV_BIN)/pip install pre-commit
	@$(VENV_BIN)/pre-commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

# Run pre-commit on all files
pre-commit: venv
	@echo "$(CYAN)Running pre-commit hooks...$(NC)"
	@if [ ! -f "$(VENV_BIN)/pre-commit" ]; then \
		echo "$(YELLOW)Pre-commit not found, installing...$(NC)"; \
		$(VENV_BIN)/pip install pre-commit; \
	fi
	@$(VENV_BIN)/pre-commit clean || true
	@$(VENV_BIN)/pre-commit run --all-files

# Validate version
validate-version: venv
	@echo "$(CYAN)Validating version...$(NC)"
	@$(VENV_BIN)/python scripts/validate_version.py

# Generate changelog
changelog: venv
	@echo "$(CYAN)Generating changelog...$(NC)"
	@$(VENV_BIN)/python scripts/generate_changelog.py
	@echo "$(GREEN)✓ Changelog generated$(NC)"

# Docker commands
docker-build:
	@echo "$(CYAN)Building Docker image...$(NC)"
	@docker build -t pyai-slayer:latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-run:
	@echo "$(CYAN)Running Docker container...$(NC)"
	@docker run --rm -it \
		-v $(PWD):/app \
		-v $(PWD)/test-results:/app/test-results \
		-v $(PWD)/reports:/app/reports \
		-e ENVIRONMENT=$(ENVIRONMENT) \
		-e HEADLESS=true \
		pyai-slayer:latest \
		/bin/bash

docker-test:
	@echo "$(CYAN)Running tests in Docker...$(NC)"
	@docker-compose --profile test run --rm test-runner
	@echo "$(GREEN)✓ Tests completed$(NC)"

docker-unit:
	@echo "$(CYAN)Running unit tests in Docker...$(NC)"
	@docker-compose --profile unit run --rm unit-tests
	@echo "$(GREEN)✓ Unit tests completed$(NC)"

docker-lint:
	@echo "$(CYAN)Running linting in Docker...$(NC)"
	@docker-compose --profile lint run --rm lint
	@echo "$(GREEN)✓ Linting completed$(NC)"

docker-shell:
	@echo "$(CYAN)Starting Docker shell...$(NC)"
	@docker-compose run --rm pyai-slayer /bin/bash

docker-clean:
	@echo "$(CYAN)Cleaning Docker resources...$(NC)"
	@docker-compose down -v
	@docker system prune -f
	@echo "$(GREEN)✓ Docker resources cleaned$(NC)"

# Metrics utilities (standalone, no Prometheus Server needed)
metrics-summary: venv
	@echo "$(CYAN)Fetching metrics summary...$(NC)"
	@$(VENV_BIN)/python scripts/metrics_utils.py --summary || \
		(echo "$(RED)Error: Metrics endpoint not available.$(NC)" && \
		 echo "$(YELLOW)Make sure:$(NC)" && \
		 echo "  1. Tests are running with enable_prometheus_metrics=True" && \
		 echo "  2. Metrics server is started (happens automatically)" && \
		 echo "  3. Port 8000 is accessible" && exit 1)

metrics-export: venv
	@echo "$(CYAN)Exporting metrics to JSON...$(NC)"
	@$(VENV_BIN)/python scripts/metrics_utils.py --export metrics.json
	@echo "$(GREEN)✓ Metrics exported to metrics.json$(NC)"

metrics-raw: venv
	@echo "$(CYAN)Fetching raw metrics...$(NC)"
	@$(VENV_BIN)/python scripts/metrics_utils.py --raw

# Dashboard commands
dashboard: venv
	@echo "$(CYAN)Starting AI Testing Dashboard...$(NC)"
	@echo "$(GREEN)Dashboard will be available at http://localhost:8080$(NC)"
	@$(VENV_BIN)/python scripts/run_dashboard.py

dashboard-custom: venv
	@echo "$(CYAN)Starting AI Testing Dashboard on custom port...$(NC)"
	@$(VENV_BIN)/python scripts/run_dashboard.py --host $(HOST) --port $(PORT)
