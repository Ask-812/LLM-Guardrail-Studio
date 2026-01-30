# Makefile for LLM Guardrail Studio

.PHONY: install test lint format clean run-dashboard run-examples help

# Default target
help:
	@echo "Available commands:"
	@echo "  install        Install dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean up generated files"
	@echo "  run-dashboard  Start Streamlit dashboard"
	@echo "  run-examples   Run example scripts"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests
test:
	python -m pytest tests/ -v --cov=guardrails --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 guardrails/ models/ dashboard/ tests/ examples/
	mypy guardrails/ models/ dashboard/

# Format code
format:
	black guardrails/ models/ dashboard/ tests/ examples/ app.py
	isort guardrails/ models/ dashboard/ tests/ examples/ app.py

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

# Run Streamlit dashboard
run-dashboard:
	streamlit run app.py

# Run examples
run-examples:
	python examples/basic_usage.py
	python examples/advanced_usage.py

# Build package
build:
	python setup.py sdist bdist_wheel

# Run quick test
quick-test:
	python -c "from guardrails import GuardrailPipeline; print('✅ Import successful')"
	python -c "from guardrails import GuardrailPipeline; p = GuardrailPipeline(); r = p.evaluate('test', 'test'); print('✅ Basic evaluation works')"