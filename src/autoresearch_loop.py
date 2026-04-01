"""Titanic Autoresearch Loop - Autonomous Feature Engineering Framework.

This module implements the autoresearch pattern for iterative feature engineering
on the Titanic dataset. An agent autonomously iterates on feature hypotheses,
training models and analyzing results to improve AUC-ROC.

Based on Karpathy's autoresearch pattern: https://github.com/karpathy/autoresearch
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
MAX_ITER: int = 1000
BASELINE_AUC: float = 0.8654
PLOT_GENERATION_INTERVAL: int = 3
MIN_LOGS_FOR_PLOTS: int = 3
FIGURE_DPI: int = 150
FIGURE_SIZE_CONVERGENCE: Tuple[int, int] = (10, 6)
FIGURE_SIZE_IMPORTANCE: Tuple[int, int] = (10, 6)
HEATMAP_MIN_HEIGHT: int = 4
HEATMAP_HEIGHT_MULTIPLIER: float = 0.4

DATA_PATH: Path = Path("data/raw/titanic_original.csv")
LOGS_PATH: Path = Path("logs/iterations.jsonl")
PLOTS_DIR: Path = Path("plots")

# Ensure directories exist
LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA PIPELINE - Baseline preprocessing
# =============================================================================


def load_raw_data() -> pd.DataFrame:
    """Load raw Titanic data with all columns available.

    Agent can derive features from any column in the dataset.

    Available columns:
        - pclass: Passenger class (1, 2, 3)
        - survived: Target variable (0 or 1)
        - name: Passenger name (for Title extraction)
        - sex: Gender (male, female)
        - age: Age in years (has ~20% missing values)
        - sibsp: Number of siblings/spouses aboard
        - parch: Number of parents/children aboard
        - ticket: Ticket number (complex, sparse)
        - fare: Ticket fare
        - cabin: Cabin number (for deck extraction, ~77% missing)
        - embarked: Port of embarkation (C, Q, S)

    Returns:
        DataFrame with raw Titanic data.

    Raises:
        FileNotFoundError: If data file does not exist.
        pd.errors.ParserError: If CSV is malformed.
    """
    try:
        logger.info(f"Loading raw data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Remove rows with missing target
        initial_rows = len(df)
        df = df.dropna(subset=["survived"])
        removed_rows = initial_rows - len(df)
        logger.info(f"Removed {removed_rows} rows with missing target: {len(df)} rows remain")
        logger.info(f"Available columns: {list(df.columns)}")

        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found at {DATA_PATH}")
        raise e
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise e


def preprocess_baseline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply baseline preprocessing: imputation and one-hot encoding.

    Handles missing values and categorical encoding to create X, y ready
    for train_test_split. This preprocessing is intentionally simple and
    non-modifiable to serve as a stable baseline.

    Args:
        df: Raw input DataFrame with all columns.

    Returns:
        Tuple of (X, y) where X is numeric features and y is target.

    Raises:
        KeyError: If expected columns are missing.
        ValueError: If preprocessing fails.
    """
    try:
        # Drop columns that are IDs or too sparse
        df = df.drop(
            columns=["passengerid", "ticket", "boat", "body", "home.dest"],
            errors="ignore",
        )

        # Extract target
        y = df["survived"].copy()
        df = df.drop(columns=["survived"])

        # Imputation: Age by Pclass median
        df["age"] = df.groupby("pclass")["age"].transform(
            lambda x: x.fillna(x.median())
        )

        # Imputation: Fare (very few missing)
        df["fare"] = df["fare"].fillna(df["fare"].median())

        # Imputation: Embarked (very few missing)
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

        # One-hot encoding for categorical variables
        df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

        # Extract numeric columns (skip name, cabin for baseline)
        X = df.select_dtypes(include=[np.number, bool]).astype(float)

        return X, y
    except KeyError as e:
        logger.error(f"Missing expected column: {e}")
        raise e
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise e


def prepare_data_for_training(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data and apply standardization.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test).

    Raises:
        ValueError: If data shapes are invalid.
    """
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        return X_tr_s, X_te_s, y_tr, y_te
    except ValueError as e:
        logger.error(f"Data preparation failed: {e}")
        raise e


def train_evaluate(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    feature_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Train LogisticRegression model and compute evaluation metrics.

    Args:
        X_tr: Training features (scaled).
        X_te: Test features (scaled).
        y_tr: Training target.
        y_te: Test target.
        feature_names: Optional list of feature names for importance tracking.

    Returns:
        Dictionary with keys: auc_roc, precision, recall, f1, model, feature_importance.

    Raises:
        ValueError: If model training fails.
    """
    try:
        model = LogisticRegression(
            random_state=RANDOM_STATE, max_iter=MAX_ITER
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, y_proba)
        pr = precision_score(y_te, y_pred)
        rc = recall_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred)

        # Generate feature importance
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


# =============================================================================
# LOGGING UTILITIES
# =============================================================================


def append_log(entry: Dict[str, Any]) -> None:
    """Append a single JSON line to the iterations log.

    Args:
        entry: Dictionary containing iteration results.

    Raises:
        IOError: If file write fails.
    """
    try:
        with open(LOGS_PATH, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"Failed to write log entry: {e}")
        raise e


def load_logs() -> list:
    """Load all previous iterations from the log file.

    Returns:
        List of iteration dictionaries, or empty list if no logs exist.

    Raises:
        json.JSONDecodeError: If log file contains invalid JSON.
    """
    try:
        if not LOGS_PATH.exists():
            return []
        with open(LOGS_PATH) as f:
            logs = []
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
            return logs
    except json.JSONDecodeError as e:
        logger.error(f"Log file contains invalid JSON: {e}")
        raise e
    except IOError as e:
        logger.error(f"Failed to read log file: {e}")
        raise e


def get_best_auc() -> float:
    """Get the best AUC from all previous iterations.

    Returns:
        Best AUC found so far, or 0.5 as default if no logs exist.
    """
    try:
        logs = load_logs()
        return max((log["metrics"]["auc_roc"] for log in logs), default=0.5)
    except Exception as e:
        logger.error(f"Failed to get best AUC: {e}")
        return 0.5


def get_current_iteration() -> int:
    """Get the next iteration number based on existing logs.

    Returns:
        Number of completed iterations (0-indexed).
    """
    try:
        logs = load_logs()
        return len(logs)
    except Exception as e:
        logger.error(f"Failed to get iteration count: {e}")
        return 0


# =============================================================================
# VISUALIZATION
# =============================================================================


def generate_plots() -> None:
    """Generate three plots summarizing iteration progress.

    Plots are saved to PLOTS_DIR:
        - convergence_auc.png: AUC trend over iterations
        - feature_importance_latest.png: Top 10 features in latest model
        - metrics_dashboard.png: Heatmap of all metrics

    Raises:
        IOError: If plot saving fails.
    """
    try:
        logs = load_logs()
        if len(logs) < MIN_LOGS_FOR_PLOTS:
            return

        logger.info("Generating visualizations...")

        # Convergence plot
        iters = [log["iteration"] for log in logs]
        aucs = [log["metrics"]["auc_roc"] for log in logs]

        plt.figure(figsize=FIGURE_SIZE_CONVERGENCE)
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
        plt.savefig(PLOTS_DIR / "convergence_auc.png", dpi=FIGURE_DPI)
        plt.close()
        logger.info("Saved convergence_auc.png")

        # Feature importance (latest)
        latest = logs[-1]
        if "feature_importance" in latest and latest["feature_importance"]:
            fi = latest["feature_importance"]
            top = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            names, vals = zip(*top)
            plt.figure(figsize=FIGURE_SIZE_IMPORTANCE)
            plt.barh(range(len(names)), vals)
            plt.yticks(range(len(names)), names)
            plt.gca().invert_yaxis()
            plt.xlabel("Coefficient")
            plt.title(f"Feature Importance - Iteration {latest['iteration']}")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "feature_importance_latest.png", dpi=FIGURE_DPI)
            plt.close()
            logger.info("Saved feature_importance_latest.png")

        # Metrics heatmap
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
        plt.figure(figsize=(8, max(HEATMAP_MIN_HEIGHT, len(logs) * HEATMAP_HEIGHT_MULTIPLIER)))
        sns.heatmap(
            df_metrics,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        plt.title("Metrics Dashboard")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "metrics_dashboard.png", dpi=FIGURE_DPI)
        plt.close()
        logger.info("Saved metrics_dashboard.png")
    except IOError as e:
        logger.error(f"Failed to save plots: {e}")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - Agent modifies create_features()
# =============================================================================


def create_features(
    X: pd.DataFrame,
    y: pd.Series,
    df_raw: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create engineered features from baseline features.

    AGENT MODIFIES THIS FUNCTION to add new feature hypotheses.

    Args:
        X: Preprocessed baseline features (numeric only).
        y: Target variable (unused in baseline).
        df_raw: Original raw dataframe (for extracting Name, Cabin, etc.).

    Returns:
        DataFrame with engineered features (numeric columns only).
    """
    X_new = X.copy()

    # ITERATION 0: Baseline - no features added
    # TODO: Agent adds feature engineering code here in next iteration

    return X_new


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    """Execute one iteration of the autoresearch loop.

    Raises:
        Exception: Any unhandled exception during execution.
    """
    try:
        logger.info("=" * 60)
        logger.info("TITANIC AUTORESEARCH - Starting")
        logger.info("=" * 60)

        # Load raw data (all columns available)
        df_raw = load_raw_data()

        # Baseline preprocessing
        X_base, y = preprocess_baseline(df_raw)
        logger.info(f"Baseline features: {list(X_base.columns)}")

        # Create features (agent modifies this function)
        X = create_features(X_base, y, df_raw=df_raw)
        feature_names = list(X.columns)

        # Train/test split and evaluate
        X_tr, X_te, y_tr, y_te = prepare_data_for_training(X, y)
        results = train_evaluate(X_tr, X_te, y_tr, y_te, feature_names)

        # Determine iteration info
        iteration = get_current_iteration()
        baseline_auc = 0.5 if iteration == 0 else get_best_auc()
        improvement = results["auc_roc"] - baseline_auc

        # Build log entry
        log_entry: Dict[str, Any] = {
            "iteration": iteration,
            "hypothesis": (
                "Baseline without feature engineering"
                if iteration == 0
                else "TODO"
            ),
            "feature_name": "baseline" if iteration == 0 else "TODO",
            "feature_code": (
                "# No feature engineering - baseline only"
                if iteration == 0
                else "TODO"
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
                results["feature_importance"]
                if results["feature_importance"]
                else {}
            ),
        }

        # Log and output results
        append_log(log_entry)
        logger.info("RESULTS:")
        logger.info(f"  AUC:       {results['auc_roc']:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  F1:        {results['f1']:.4f}")
        logger.info(
            f"  Improvement: {improvement:+.4f} vs baseline ({baseline_auc:.4f})"
        )
        logger.info(f"Logged iteration {iteration}")

        # Generate plots every PLOT_GENERATION_INTERVAL iterations
        if iteration % PLOT_GENERATION_INTERVAL == 0 and iteration >= MIN_LOGS_FOR_PLOTS:
            generate_plots()

        logger.info("=" * 60)
        if iteration == 0:
            logger.info(
                "Iteration 0 (Baseline) - Agent should modify create_features() next"
            )
        else:
            logger.info(f"Iteration {iteration} complete. Continue with next hypothesis.")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
