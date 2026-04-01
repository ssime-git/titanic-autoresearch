"""Interpretability visualizations for model analysis."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)


def plot_feature_coefficients(
    coefficients: Dict[str, float], output_path: Path, top_n: int = 15
) -> None:
    """Plot feature coefficients with positive/negative highlighting.

    Args:
        coefficients: Feature importance from model.
        output_path: Path to save figure.
        top_n: Number of top features to display.
    """
    try:
        sorted_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:top_n]

        features, coefs = zip(*top_features)
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coefs]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))

        ax.barh(y_pos, coefs, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel("Coefficient Value", fontsize=11)
        ax.set_title("Feature Coefficients (Top 15)\nGreen=Positive (increases survival), Red=Negative",
                     fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved feature coefficients plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot feature coefficients: {e}")


def plot_prediction_distributions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Plot prediction probability distributions by actual outcome.

    Args:
        y_true: True labels (0/1).
        y_pred_proba: Predicted probabilities for positive class.
        output_path: Path to save figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pred_proba_list = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba

        ax.hist(
            y_pred_proba_list[y_true == 0],
            bins=30,
            alpha=0.6,
            label="Did Not Survive (y=0)",
            color="#e74c3c",
        )
        ax.hist(
            y_pred_proba_list[y_true == 1],
            bins=30,
            alpha=0.6,
            label="Survived (y=1)",
            color="#2ecc71",
        )

        ax.set_xlabel("Predicted Probability of Survival", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Prediction Probability Distributions by Outcome", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved prediction distributions plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot prediction distributions: {e}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels (thresholded at 0.5).
        output_path: Path to save figure.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"],
        )

        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")

        fig.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved confusion matrix plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    auc_score: float,
    output_path: Path,
) -> None:
    """Plot ROC curve.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for positive class.
        auc_score: AUC-ROC score.
        output_path: Path to save figure.
    """
    try:
        y_pred_proba_list = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba_list)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
        ax.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curve (Test Set)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved ROC curve plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot ROC curve: {e}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Plot precision-recall curve.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for positive class.
        output_path: Path to save figure.
    """
    try:
        y_pred_proba_list = y_pred_proba.flatten() if y_pred_proba.ndim > 1 else y_pred_proba
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba_list)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(recall, precision, color="#9b59b6", lw=2.5, label="Precision-Recall Curve")
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Precision-Recall Curve (Test Set)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        fig.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved precision-recall curve plot to {output_path}")
    except Exception as e:
        logger.error(f"Failed to plot precision-recall curve: {e}")
