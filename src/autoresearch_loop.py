"""
Titanic Autoresearch Loop - Agent Framework
The agent modifies ONLY this file to add new hypothesis iterations.
Based on Karpathy's autoresearch pattern.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
DATA_PATH = Path("data/raw/titanic_original.csv")
LOGS_PATH = Path("logs/iterations.jsonl")
PLOTS_DIR = Path("plots")

# Ensure directories exist
LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA PIPELINE - Baseline preprocessing (DO NOT MODIFY)
# =============================================================================

def load_raw_data():
    """
    Load raw Titanic data with all columns available.
    Agent can derive features from any column.

    Available columns:
    - pclass: Passenger class (1, 2, 3)
    - survived: Target variable (0 or 1)
    - name: Passenger name (for Title extraction)
    - sex: Gender (male, female)
    - age: Age in years (has missing values)
    - sibsp: Number of siblings/spouses aboard
    - parch: Number of parents/children aboard
    - ticket: Ticket number (complex, sparse)
    - fare: Ticket fare
    - cabin: Cabin number (for deck extraction, mostly missing)
    - embarked: Port of embarkation (C, Q, S)
    """
    print("📊 Loading raw data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Remove rows with missing target
    df = df.dropna(subset=["survived"])
    print(f"   After removing missing target: {len(df)} rows")
    print(f"   Columns available: {list(df.columns)}")

    return df


def preprocess_baseline(df):
    """
    Baseline preprocessing - handles missing values and categorical encoding.
    Creates X, y ready for train_test_split.
    """
    # Drop columns that are IDs or too sparse
    df = df.drop(columns=["passengerid", "ticket", "boat", "body", "home.dest"], errors="ignore")

    # Extract target
    y = df["survived"].copy()
    df = df.drop(columns=["survived"])

    # Imputation: Age by Pclass median
    df["age"] = df.groupby("pclass")["age"].transform(lambda x: x.fillna(x.median()))

    # Imputation: Fare
    df["fare"] = df["fare"].fillna(df["fare"].median())

    # Imputation: Embarked
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

    # Extract numeric columns (skip name, cabin for baseline - agent can use them)
    X = df.select_dtypes(include=[np.number, bool]).astype(float)

    return X, y


def prepare_data_for_training(X, y):
    """Train/test split + standardization"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_te_s, y_tr, y_te


def train_evaluate(X_tr, X_te, y_tr, y_te, feature_names=None):
    """Train LogisticRegression and return metrics"""
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, y_proba)
    pr = precision_score(y_te, y_pred)
    rc = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)

    return {
        "auc_roc": float(auc),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "model": model,
        "feature_importance": dict(
            zip(
                feature_names or [f"f_{i}" for i in range(X_tr.shape[1])],
                model.coef_[0].tolist(),
            )
        ),
    }


# =============================================================================
# LOGGING UTILITIES
# =============================================================================


def append_log(entry):
    """Append JSON line to logs/iterations.jsonl"""
    with open(LOGS_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_logs():
    """Load all previous iterations from logs"""
    if not LOGS_PATH.exists():
        return []
    with open(LOGS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_best_auc():
    """Get best AUC from previous iterations"""
    logs = load_logs()
    return max((log["metrics"]["auc_roc"] for log in logs), default=0.5)


def get_current_iteration():
    """Get next iteration number"""
    logs = load_logs()
    return len(logs)


# =============================================================================
# VISUALIZATION
# =============================================================================


def generate_plots():
    """Generate plots after 3+ iterations"""
    logs = load_logs()
    if len(logs) < 3:
        return

    print("\n📊 Generating visualizations...")

    # Convergence plot
    iters = [log["iteration"] for log in logs]
    aucs = [log["metrics"]["auc_roc"] for log in logs]

    plt.figure(figsize=(10, 6))
    plt.plot(iters, aucs, "bo-")
    plt.axhline(y=0.8654, color="gray", linestyle="--", label="Baseline (0.8654)")
    plt.xlabel("Iteration")
    plt.ylabel("AUC-ROC")
    plt.title("Feature Engineering Convergence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "convergence_auc.png", dpi=150)
    plt.close()
    print("   Saved convergence_auc.png")

    # Feature importance (latest)
    latest = logs[-1]
    if "feature_importance" in latest and latest["feature_importance"]:
        fi = latest["feature_importance"]
        top = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        names, vals = zip(*top)
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(names)), vals)
        plt.yticks(range(len(names)), names)
        plt.gca().invert_yaxis()
        plt.xlabel("Coefficient")
        plt.title(f"Feature Importance - Iter {latest['iteration']}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance_latest.png", dpi=150)
        plt.close()
        print("   Saved feature_importance_latest.png")

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
    plt.figure(figsize=(8, max(4, len(logs) * 0.4)))
    sns.heatmap(df_metrics, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Metrics Dashboard")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_dashboard.png", dpi=150)
    plt.close()
    print("   Saved metrics_dashboard.png")


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - Agent adds/modifies these
# =============================================================================


def create_features(X, y, df_raw=None):
    """
    AGENT MODIFIES THIS FUNCTION
    Current: baseline (no feature engineering)
    Next: add your hypothesis feature here

    Args:
        X: Preprocessed features (numeric only)
        y: Target variable
        df_raw: Original raw dataframe (for extraction of Name, Cabin, etc.)

    Returns:
        X_new: DataFrame with features (numeric)
    """
    X_new = X.copy()

    # =============================================
    # ITERATION 0: Baseline - no features added
    # =============================================
    # TODO: Agent adds feature code here in next iteration

    return X_new


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main experiment loop"""
    print("=" * 60)
    print("TITANIC AUTORESEARCH - Starting")
    print("=" * 60)

    # Load raw data (all columns available)
    df_raw = load_raw_data()

    # Baseline preprocessing
    X_base, y = preprocess_baseline(df_raw)

    print(f"\n📋 Baseline features: {list(X_base.columns)}")

    # Create features (agent modifies this)
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
    log_entry = {
        "iteration": iteration,
        "hypothesis": "Baseline sans feature engineering" if iteration == 0 else "TODO",
        "feature_name": "baseline" if iteration == 0 else "TODO",
        "feature_code": "# No feature engineering - baseline only" if iteration == 0 else "TODO",
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
        "feature_importance": results["feature_importance"] if results["feature_importance"] else {},
    }

    # Log and output
    append_log(log_entry)
    print(f"\n📈 Results:")
    print(f"   AUC:       {results['auc_roc']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1:        {results['f1']:.4f}")
    print(f"   Improvement: {improvement:+.4f} vs baseline ({baseline_auc:.4f})")
    print(f"\n📝 Logged iteration {iteration}")

    # Generate plots every 3 iterations
    if iteration % 3 == 0 and iteration >= 3:
        generate_plots()

    print("\n" + "=" * 60)
    if iteration == 0:
        print("Iteration 0 (Baseline) - Agent should modify create_features() next")
    else:
        print(f"Iteration {iteration} complete. Continue with next hypothesis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
