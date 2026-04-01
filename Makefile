# Titanic Autoresearch Makefile
# Uses UV (ultrafast Python package manager) for dependency management

.PHONY: setup install download run clean check help

# Default target
.DEFAULT_GOAL := all

# Main setup
all: install download

# Install UV if not present and create venv with dependencies
install:
	@echo "🔧 Installing UV and dependencies..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "📦 Installing UV..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		export PATH="$$HOME/.cargo/bin:$$PATH"; \
	fi
	@echo "📦 Creating venv and installing dependencies..."
	@uv venv --seed
	@uv pip install -r requirements.txt
	@echo "✅ Environment ready. Activate with: source .venv/bin/activate"

# Quick install using uvx (no venv needed, runs directly)
uvx-install:
	@echo "📦 UVX mode - dependencies will be installed on first run"
	@echo "✅ Ready to use 'make run-uvx'"

# Download dataset
download:
	@echo "📥 Downloading Titanic dataset..."
	@mkdir -p data/raw
	@uvx python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv', 'data/raw/titanic_original.csv')" 2>/dev/null || \
	python3 -c "import requests; open('data/raw/titanic_original.csv','wb').write(requests.get('https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv').content)" 2>/dev/null || \
	curl -L -o data/raw/titanic_original.csv https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv
	@echo "✅ Dataset downloaded"

# Run with activated venv
run:
	@echo "🚀 Running autoresearch loop..."
	@. .venv/bin/activate && python src/autoresearch_loop.py

# Run with UVX (no venv needed - pulls deps automatically)
run-uvx:
	@echo "🚀 Running with UVX..."
	@uvx --python 3.10 \
		--with pandas>=1.5.0 \
		--with scikit-learn>=1.2.0 \
		--with matplotlib>=3.6.0 \
		--with seaborn>=0.12.0 \
		--with numpy>=1.23.0 \
		python src/autoresearch_loop.py

# Run with UV (using project venv)
	@echo "🚀 Running with UV..."
	@uv run python src/autoresearch_loop.py

# Check environment
check:
	@echo "🔍 Checking setup..."
	@command -v uv >/dev/null 2>&1 && echo "✅ UV installed" || echo "❌ UV not found"
	@test -d .venv && echo "✅ Virtual env exists" || echo "⚠️  No venv (use 'make install')"
	@test -f data/raw/titanic_original.csv && echo "✅ Dataset ready" || echo "❌ Dataset missing (use 'make download')"

# Clean outputs (keep venv and raw data)
clean:
	@echo "🧹 Cleaning outputs..."
	@rm -rf data/processed/* logs/* plots/* __pycache__ .pytest_cache *.pyc
	@mkdir -p data/processed logs plots
	@echo "✅ Cleaned"

# Deep clean (remove venv too)
clean-all: clean
	@echo "🧹 Removing virtual environment..."
	@rm -rf .venv
	@echo "✅ Fully cleaned"

# Help
help:
	@echo "Titanic Autoresearch - UV/UVX Ready"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Create venv with UV and install deps"
	@echo "  make uvx-install  - Skip venv, use UVX directly"
	@echo "  make download     - Download Titanic dataset"
	@echo "  make all          - Install + download (default)"
	@echo ""
	@echo "Run:"
	@echo "  make run          - Run with venv (after make install)"
	@echo "  make run-uv       - Run with 'uv run'"
	@echo "  make run-uvx      - Run with UVX (no venv needed)"
	@echo ""
	@echo "Other:"
	@echo "  make check        - Verify setup"
	@echo "  make clean        - Remove outputs only"
	@echo "  make clean-all    - Remove everything including venv"
	@echo "  make help         - Show this help"
