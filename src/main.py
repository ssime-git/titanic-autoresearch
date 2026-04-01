"""Main orchestration module for autoresearch loop."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .data import load_raw_data, preprocess_baseline, prepare_data_for_training
from .features import create_features
from .logging_utils import append_log, get_best_auc, get_current_iteration
from .model import train_evaluate
from .visualization import generate_plots

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 1000
BASELINE_AUC = 0.8654
PLOT_GENERATION_INTERVAL = 3
MIN_LOGS_FOR_PLOTS = 3


def main(
    data_path: Path,
    logs_path: Path,
    plots_dir: Path,
) -> None:
    """Execute one iteration of the autoresearch loop.

    Args:
        data_path: Path to raw data CSV.
        logs_path: Path to iteration logs.
        plots_dir: Directory for visualizations.

    Raises:
        Exception: Any unhandled exception during execution.
    """
    try:
        logger.info("=" * 60)
        logger.info("TITANIC AUTORESEARCH - Starting")
        logger.info("=" * 60)

        df_raw = load_raw_data(data_path)
        X_base, y = preprocess_baseline(df_raw)
        logger.info(f"Baseline features: {list(X_base.columns)}")

        X = create_features(X_base, y, df_raw=df_raw)
        feature_names = list(X.columns)

        X_tr, X_te, y_tr, y_te = prepare_data_for_training(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        results = train_evaluate(
            X_tr,
            X_te,
            y_tr,
            y_te,
            feature_names=feature_names,
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
        )

        iteration = get_current_iteration(logs_path)
        baseline_auc = 0.5 if iteration == 0 else get_best_auc(logs_path)
        improvement = results["auc_roc"] - baseline_auc

        log_entry: Dict[str, Any] = {
            "iteration": iteration,
            "hypothesis": (
                "Baseline without feature engineering" if iteration == 0 else "TODO"
            ),
            "feature_name": "baseline" if iteration == 0 else "TODO",
            "feature_code": (
                "# No feature engineering - baseline only" if iteration == 0 else "TODO"
            ),
            "metrics": {
                "auc_roc": round(results["auc_roc"], 4),
                "precision": round(results["precision"], 4),
                "recall": round(results["recall"], 4),
                "f1": round(results["f1"], 4),
            },
            "baseline_auc": baseline_auc,
            "improvement": f"{improvement:+.4f}",
            "status": "baseline" if iteration == 0 else "TODO",
            "analysis": f"Iteration {iteration} complete. AUC={results['auc_roc']:.4f}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "feature_importance": (
                results["feature_importance"] if results["feature_importance"] else {}
            ),
        }

        append_log(log_entry, logs_path)
        logger.info("RESULTS:")
        logger.info(f"  AUC:       {results['auc_roc']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  F1:        {results['f1']:.4f}")
        logger.info(
            f"  Improvement: {improvement:+.4f} vs baseline ({baseline_auc:.4f})"
        )
        logger.info(f"Logged iteration {iteration}")

        if (
            iteration % PLOT_GENERATION_INTERVAL == 0
            and iteration >= MIN_LOGS_FOR_PLOTS
        ):
            generate_plots(logs_path, plots_dir)

        logger.info("=" * 60)
        if iteration == 0:
            logger.info(
                "Iteration 0 (Baseline) - Agent should modify create_features() next"
            )
        else:
            logger.info(
                f"Iteration {iteration} complete. Continue with next hypothesis."
            )
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    DATA_PATH = Path("data/raw/titanic_original.csv")
    LOGS_PATH = Path("logs/iterations.jsonl")
    PLOTS_DIR = Path("plots")

    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    main(DATA_PATH, LOGS_PATH, PLOTS_DIR)
