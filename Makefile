# Titanic Autoresearch Makefile
# Uses UVX for dependency management (no venv needed)

.PHONY: setup install download run check clean clean-all help

# Default target
.DEFAULT_GOAL := help

# Full setup: install dependencies and download data
setup: download
	@echo "✅ Setup complete. Run 'make run' to start autoresearch loop."

# Download dataset only
download:
	@echo "📥 Downloading Titanic dataset from Geoyi..."
	@mkdir -p data/raw
	@test -f data/raw/titanic_original.csv && echo "✅ Dataset already exists" || \
	curl -L -o data/raw/titanic_original.csv "https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv" && \
	echo "✅ Dataset downloaded"

# Run autoresearch loop with UVX (recommended - no venv setup needed)
run: download
	@echo "🚀 Running Titanic Autoresearch Loop with UVX..."
	@uvx --python 3.10 \
		--with pandas \
		--with scikit-learn \
		--with matplotlib \
		--with seaborn \
		--with numpy \
		python src/autoresearch_loop.py

# Check environment and data
check:
	@echo "🔍 Checking setup..."
	@command -v uvx >/dev/null 2>&1 && echo "✅ UVX available" || echo "❌ UVX not found (install Anthropic SDK)"
	@test -f data/raw/titanic_original.csv && echo "✅ Dataset ready" || echo "⚠️  Dataset missing (run 'make download')"
	@test -d logs && echo "✅ Logs directory exists" || mkdir -p logs && echo "✅ Created logs directory"
	@test -d plots && echo "✅ Plots directory exists" || mkdir -p plots && echo "✅ Created plots directory"

# Clean outputs (keep raw data and src code)
clean:
	@echo "🧹 Cleaning outputs..."
	@rm -rf logs/*.jsonl plots/*.png data/processed/* __pycache__ .pytest_cache *.pyc 2>/dev/null || true
	@mkdir -p logs plots data/processed
	@echo "✅ Cleaned (kept data/raw and src)"

# Deep clean (remove everything except src and raw data)
clean-all: clean
	@echo "🧹 Deep clean..."
	@echo "✅ Outputs removed"

# Help
help:
	@echo "Titanic Autoresearch - Autoresearch Loop with Autonomous Agent"
	@echo ""
	@echo "Usage:"
	@echo "  make setup        - Download dataset (full one-time setup)"
	@echo "  make run          - Run autoresearch loop (uses UVX, no venv needed)"
	@echo "  make check        - Verify setup is ready"
	@echo "  make clean        - Remove iteration outputs only (keep data)"
	@echo "  make clean-all    - Remove all outputs"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make setup     # Download data once"
	@echo "  2. make run       # Start autoresearch loop"
	@echo ""
	@echo "To give to an agent:"
	@echo "  See README.md for the complete agent prompt"
