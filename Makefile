# Makefile for Active Learning Classifier

.PHONY: help install install-dev test lint format clean run docker-build docker-run

help:
	@echo "Active Learning Classifier - Development Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install package dependencies"
	@echo "  make install-dev    - Install package with dev dependencies"
	@echo "  make test           - Run tests with pytest"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black"
	@echo "  make type-check     - Run type checking with mypy"
	@echo "  make clean          - Remove build artifacts and cache"
	@echo "  make run            - Run Streamlit application"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-compose - Run with docker-compose"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ app.py --max-line-length=100
	black --check src/ tests/ app.py

format:
	black src/ tests/ app.py

type-check:
	mypy src/ --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

run:
	streamlit run app.py

benchmark:
	python scripts/benchmark_comparison.py

docker-build:
	docker build -t active-learning-classifier:latest .

docker-run:
	docker run -p 8501:8501 active-learning-classifier:latest

docker-compose:
	docker-compose up -d

docker-stop:
	docker-compose down

# CI/CD commands
ci-test: install-dev lint type-check test-cov
	@echo "All CI checks passed!"

# Setup development environment
setup-dev: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make run' to start the application"
