"""Model training and evaluation module."""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def train_evaluate(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    feature_names: Optional[list] = None,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """Train LogisticRegression model and compute evaluation metrics.

    Args:
        X_tr: Training features (scaled).
        X_te: Test features (scaled).
        y_tr: Training target.
        y_te: Test target.
        feature_names: Optional list of feature names for importance tracking.
        random_state: Random seed for reproducibility.
        max_iter: Maximum iterations for LogisticRegression.

    Returns:
        Dictionary with keys: auc_roc, precision, recall, f1, model, feature_importance.

    Raises:
        ValueError: If model training fails.
    """
    try:
        model = LogisticRegression(random_state=random_state, max_iter=max_iter)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, y_proba)
        pr = precision_score(y_te, y_pred)
        rc = recall_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)

        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(X_tr.shape[1])]

        return {
            "auc_roc": float(auc),
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "model": model,
            "feature_importance": dict(zip(feature_names, model.coef_[0].tolist())),
        }
    except ValueError as e:
        logger.error(f"Model training failed: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise e
