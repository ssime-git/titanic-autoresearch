"""Analyze and visualize the final optimized model."""

import logging
from pathlib import Path

from .data import load_raw_data, preprocess_baseline, prepare_data_for_training
from .features import create_features
from .model import train_evaluate
from .interpretability import (
    plot_feature_coefficients,
    plot_prediction_distributions,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 1000


def analyze_final_model(
    data_path: Path,
    plots_dir: Path,
) -> None:
    """Generate interpretability visualizations for the final optimized model.

    Args:
        data_path: Path to raw data CSV.
        plots_dir: Directory for visualizations.
    """
    try:
        logger.info("=" * 60)
        logger.info("FINAL MODEL INTERPRETABILITY ANALYSIS")
        logger.info("=" * 60)

        df_raw = load_raw_data(data_path)
        X_base, y = preprocess_baseline(df_raw)
        X = create_features(X_base, y, df_raw=df_raw)

        X_tr, X_te, y_tr, y_te = prepare_data_for_training(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        results = train_evaluate(
            X_tr,
            X_te,
            y_tr,
            y_te,
            feature_names=list(X.columns),
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
        )

        logger.info(f"AUC-ROC: {results['auc_roc']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1: {results['f1']:.4f}")

        auc_score = results["auc_roc"]
        feature_importance = results["feature_importance"] or {}
        y_pred_proba = results.get("y_pred_proba")
        y_pred = results.get("y_pred")

        if feature_importance:
            plot_feature_coefficients(
                feature_importance,
                plots_dir / "interpretability_coefficients.png",
            )

        if y_pred_proba is not None:
            plot_prediction_distributions(
                y_te.values,
                y_pred_proba,
                plots_dir / "interpretability_distributions.png",
            )

        if y_pred is not None:
            plot_confusion_matrix(
                y_te.values,
                y_pred,
                plots_dir / "interpretability_confusion_matrix.png",
            )

        if y_pred_proba is not None:
            plot_roc_curve(
                y_te.values,
                y_pred_proba,
                auc_score,
                plots_dir / "interpretability_roc_curve.png",
            )

            plot_precision_recall_curve(
                y_te.values,
                y_pred_proba,
                plots_dir / "interpretability_precision_recall.png",
            )

        logger.info("=" * 60)
        logger.info("Interpretability analysis complete")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error in final model analysis: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    DATA_PATH = Path("data/raw/titanic_original.csv")
    PLOTS_DIR = Path("plots")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    analyze_final_model(DATA_PATH, PLOTS_DIR)
