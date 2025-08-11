# Development Makefile for code quality tools

.PHONY: help format lint test quality install clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies including dev dependencies
	uv sync --group dev

format: ## Format code with black and isort
	python scripts/format.py

lint: ## Run linting tools (flake8, mypy)
	python scripts/lint.py

test: ## Run all tests
	python scripts/test.py

quality: ## Run all quality checks (format, lint, test)
	python scripts/quality.py

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Quick development commands
fmt: format ## Alias for format
check: quality ## Alias for quality