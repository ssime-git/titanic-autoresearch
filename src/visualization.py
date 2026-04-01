"""Visualization and plotting module."""

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .logging_utils import load_logs

logger = logging.getLogger(__name__)

BASELINE_AUC = 0.8654
PLOT_DPI = 150
MIN_LOGS_FOR_PLOTS = 3


def generate_plots(logs_path: Path, plots_dir: Path) -> None:
    """Generate three plots summarizing iteration progress.

    Args:
        logs_path: Path to the log file.
        plots_dir: Directory to save plots.

    Raises:
        IOError: If plot saving fails.
    """
    try:
        logs = load_logs(logs_path)
        if len(logs) < MIN_LOGS_FOR_PLOTS:
            return

        logger.info("Generating visualizations...")

        _plot_convergence(logs, plots_dir)
        _plot_feature_importance(logs, plots_dir)
        _plot_metrics_dashboard(logs, plots_dir)
    except IOError as e:
        logger.error(f"Failed to save plots: {e}")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")


def _plot_convergence(logs: List, plots_dir: Path) -> None:
    """Generate AUC convergence plot.

    Args:
        logs: List of iteration dictionaries.
        plots_dir: Directory to save plot.
    """
    iters = [log["iteration"] for log in logs]
    aucs = [log["metrics"]["auc_roc"] for log in logs]

    plt.figure(figsize=(10, 6))
    plt.plot(iters, aucs, "bo-")
    plt.axhline(
        y=BASELINE_AUC,
        color="gray",
        linestyle="--",
        label=f"Baseline ({BASELINE_AUC})",
    )
    plt.xlabel("Iteration")
    plt.ylabel("AUC-ROC")
    plt.title("Feature Engineering Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "convergence_auc.png", dpi=PLOT_DPI)
    plt.close()
    logger.info("Saved convergence_auc.png")


def _plot_feature_importance(logs: List, plots_dir: Path) -> None:
    """Generate feature importance plot for latest iteration.

    Args:
        logs: List of iteration dictionaries.
        plots_dir: Directory to save plot.
    """
    latest = logs[-1]
    if "feature_importance" not in latest or not latest["feature_importance"]:
        return

    fi = latest["feature_importance"]
    top = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    names, vals = zip(*top)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.title(f"Feature Importance - Iteration {latest['iteration']}")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance_latest.png", dpi=PLOT_DPI)
    plt.close()
    logger.info("Saved feature_importance_latest.png")


def _plot_metrics_dashboard(logs: List, plots_dir: Path) -> None:
    """Generate metrics dashboard heatmap.

    Args:
        logs: List of iteration dictionaries.
        plots_dir: Directory to save plot.
    """
    df_metrics = pd.DataFrame(
        [
            {
                "i": log["iteration"],
                "AUC": log["metrics"]["auc_roc"],
                "P": log["metrics"]["precision"],
                "R": log["metrics"]["recall"],
                "F1": log["metrics"]["f1"],
            }
            for log in logs
        ]
    ).set_index("i")

    plt.figure(figsize=(8, max(4, len(logs) * 0.4)))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Metrics Dashboard")
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_dashboard.png", dpi=PLOT_DPI)
    plt.close()
    logger.info("Saved metrics_dashboard.png")
