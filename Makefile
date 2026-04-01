# Titanic Autoresearch Makefile
# Uses UVX for dependency management (no venv needed)

.PHONY: setup install download run check lint format clean clean-all help

# Default target
.DEFAULT_GOAL := help

# Full setup: install dependencies and download data
setup: download
	@echo "Setup complete. Run 'make run' to start autoresearch loop."

# Download dataset only
download:
	@echo "Downloading Titanic dataset from Geoyi..."
	@mkdir -p data/raw
	@test -f data/raw/titanic_original.csv && echo "Dataset already exists" || \
	curl -L -o data/raw/titanic_original.csv "https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv" && \
	echo "Dataset downloaded"

# Run autoresearch loop with UVX (recommended - no venv setup needed)
run: download
	@echo "Running Titanic Autoresearch Loop..."
	@uvx --python 3.10 \
		--with pandas \
		--with scikit-learn \
		--with matplotlib \
		--with seaborn \
		--with numpy \
		python -m src.main

# Check environment and data
check:
	@echo "Checking setup..."
	@command -v uvx >/dev/null 2>&1 && echo "OK: UVX available" || echo "FAIL: UVX not found"
	@test -f data/raw/titanic_original.csv && echo "OK: Dataset ready" || echo "WARN: Dataset missing"
	@test -d logs && echo "OK: Logs directory exists" || mkdir -p logs && echo "OK: Created logs directory"
	@test -d plots && echo "OK: Plots directory exists" || mkdir -p plots && echo "OK: Created plots directory"

# Code quality check with ruff
lint:
	@echo "Running ruff check on src/..."
	@uvx ruff check src/ || (echo "Ruff check failed"; exit 1)
	@echo "All checks passed"

# Code format with ruff
format:
	@echo "Formatting code with ruff..."
	@uvx ruff format src/
	@echo "Formatting complete"

# Clean outputs (keep raw data and src code)
clean:
	@echo "Cleaning outputs..."
	@rm -rf logs/*.jsonl plots/*.png data/processed/* __pycache__ .pytest_cache *.pyc 2>/dev/null || true
	@mkdir -p logs plots data/processed
	@echo "Cleaned (kept data/raw and src)"

# Deep clean (remove everything except src and raw data)
clean-all: clean
	@echo "Deep clean complete"

# Help
help:
	@echo "Titanic Autoresearch - Modular Autonomous Feature Engineering"
	@echo ""
	@echo "Setup & Execution:"
	@echo "  make setup        - Download dataset (one-time setup)"
	@echo "  make run          - Run autoresearch loop via UVX"
	@echo "  make check        - Verify environment and data"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Run ruff check on src/"
	@echo "  make format       - Format code with ruff"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove iteration outputs (keep data)"
	@echo "  make clean-all    - Remove all outputs"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make setup     - Download data"
	@echo "  2. make lint      - Check code quality"
	@echo "  3. make run       - Start autoresearch loop"
