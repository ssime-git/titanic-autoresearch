# Titanic Autoresearch

Autonomous feature engineering loop using the autoresearch pattern (Karpathy).

An agent autonomously iterates on feature engineering hypotheses, testing each, analyzing results, and refining the next hypothesis. Demonstrates autonomous AI research workflows on the Titanic dataset.

## Quick Start

```bash
make setup      # Download data (one-time)
make run        # Run autoresearch loop
make check      # Verify environment and data
make lint       # Check code quality with ruff
```

## How It Works

1. Iteration 0: Baseline model without feature engineering (AUC 0.8654)
2. Iterations 1+: Agent proposes features, implements, trains, evaluates
3. Stop condition: AUC plateaus for 3 iterations OR 20 iterations reached
4. Output: `logs/iterations.jsonl` (append-only log), visualizations in `plots/`

## For Agents

To run this project with an autonomous agent:

1. Read AGENT.md for the complete prompt
2. Create a new branch: `git checkout -b autoresearch/apr1_v1`
3. Run: `make setup` then `make run`
4. Modify only `src/features.py` to add features

See AGENT.md for:
- Complete instructions and rules
- Log format specification
- Feature engineering guidelines
- Example iteration workflow

## Project Structure

```
titanic-autoresearch/
├── README.md              # This file
├── AGENT.md               # Agent prompt and instructions
├── program.md             # Detailed operational guide
├── src/                   # Modular source code
│   ├── __init__.py
│   ├── main.py           # Entry point (orchestration)
│   ├── data.py           # Load and preprocess
│   ├── model.py          # Train and evaluate
│   ├── model_config.py   # Agent modifies for model selection
│   ├── features.py       # Agent modifies for features
│   ├── logging_utils.py  # Iteration logging
│   ├── visualization.py  # Iteration progress plots
│   ├── interpretability.py # Model interpretability visualizations
│   └── analyze_final_model.py # Generate interpretability analysis
├── data/
│   ├── raw/
│   │   └── titanic_original.csv
│   └── processed/
├── logs/
│   └── iterations.jsonl
├── plots/
│   ├── convergence_auc.png
│   ├── feature_importance_latest.png
│   ├── metrics_dashboard.png
│   ├── interpretability_coefficients.png
│   ├── interpretability_distributions.png
│   ├── interpretability_confusion_matrix.png
│   ├── interpretability_roc_curve.png
│   └── interpretability_precision_recall.png
├── Makefile
├── requirements.txt
└── .gitignore
```

## Commands

```bash
make setup      # Download dataset (one-time)
make run        # Run autoresearch loop
make check      # Verify setup (environment, data, directories)
make lint       # Check code quality with ruff
make clean      # Remove iteration outputs (keep data)
make clean-all  # Remove all outputs
make help       # Show help
```

## References

- Autoresearch pattern: https://github.com/karpathy/autoresearch
- Titanic dataset: https://github.com/Geoyi/Cleaning-Titanic-Data

## License

Demonstration project for autonomous AI research workflows.
