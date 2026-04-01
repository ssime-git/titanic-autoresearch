"""
Titanic Autoresearch Loop - Agent Framework
The agent modifies ONLY this file to add new hypothesis iterations.
Based on Karpathy's autoresearch pattern.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
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

def load_and_preprocess_data():
    """
    Baseline preprocessing - NON MODIFIABLE par l'agent
    1. Load CSV
    2. Suppression: PassengerId, Ticket, Cabin (trop sparse)
    3. Imputation Age -> median par Pclass
    4. One-hot encoding: Sex, Embarked
    5. Output: X, y pret pour train_test_split
    """
    print("📊 Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {len(df)} rows")

    # Normalize column names to lowercase (Geoyi dataset uses lowercase)
    df.columns = df.columns.str.lower()

    # Remove rows with missing target
    df = df.dropna(subset=["survived"])
    print(f"   After removing missing target: {len(df)} rows")

    # Suppression colonnes non utiles (sparse or non-predictive)
    df = df.drop(columns=["passengerid", "ticket", "cabin", "boat", "body", "home.dest", "name"], errors="ignore")

    # Target
    y = df["survived"].copy()
    df = df.drop(columns=["survived"])

    # Imputation Age
    df["age"] = df.groupby("pclass")["age"].transform(lambda x: x.fillna(x.median()))

    # Imputation Fare (very few missing)
    df["fare"] = df["fare"].fillna(df["fare"].median())

    # Imputation Embarked (very few missing)
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # One-hot encoding
    df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

    # Keep only numeric and boolean (from one-hot encoding)
    X = df.select_dtypes(include=[np.number, bool])
    X = X.astype(float)  # Convert bool to float for ML
    print(f"   Features: {list(X.columns)}")
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
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0
    
    return {
        "auc_roc": round(auc, 4),
        "precision": round(pr, 4),
        "recall": round(rc, 4),
        "f1": round(f1, 4),
        "model": model,
        "feature_importance": dict(zip(
            feature_names or [f"f_{i}" for i in range(X_tr.shape[1])],
            model.coef_[0].tolist()
        ))
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
    plt.axhline(y=logs[0]["metrics"]["auc_roc"], color="gray", linestyle="--", label="Baseline")
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
    if "feature_importance" in latest:
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
    df = pd.DataFrame([
        {"i": log["iteration"], "AUC": log["metrics"]["auc_roc"],
         "P": log["metrics"]["precision"], "R": log["metrics"]["recall"],
         "F1": log["metrics"]["f1"]} for log in logs
    ]).set_index("i")
    plt.figure(figsize=(8, max(4, len(logs)*0.4)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Metrics Dashboard")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_dashboard.png", dpi=150)
    plt.close()
    print("   Saved metrics_dashboard.png")


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - Agent adds/modifies these
# =============================================================================

def create_features(X, y):
    """
    AGENT MODIFIES THIS FUNCTION
    Current: baseline (no feature engineering)
    Next: add your hypothesis feature here
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
    print("="*60)
    print("TITANIC AUTORESEARCH - Starting")
    print("="*60)
    
    # Load data
    X_base, y = load_and_preprocess_data()
    
    # Create features (agent modifies this)
    X = create_features(X_base, y)
    feature_names = list(X.columns)
    
    # Train/test split and evaluate
    X_tr, X_te, y_tr, y_te = prepare_data_for_training(X, y)
    results = train_evaluate(X_tr, X_te, y_tr, y_te, feature_names)
    
    # Determine iteration info
    iteration = get_current_iteration()
    baseline_auc = get_best_auc() if iteration > 0 else 0.5
    improvement = results["auc_roc"] - baseline_auc
    
    # Build log entry
    log_entry = {
        "iteration": iteration,
        "hypothesis": "Baseline sans feature engineering",
        "feature_name": "baseline",
        "feature_code": "# No feature engineering - baseline only",
        "metrics": {
            "auc_roc": results["auc_roc"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"]
        },
        "baseline_auc": baseline_auc,
        "improvement": f"{improvement:+.4f}",
        "status": "keep",
        "analysis": f"Iteration {iteration} complete. AUC={results['auc_roc']:.4f}",
        "timestamp": datetime.now().isoformat() + "Z",
        "feature_importance": results["feature_importance"]
    }
    
    # Log and output
    append_log(log_entry)
    print(f"\n📈 Results:")
    print(f"   AUC:     {results['auc_roc']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1:        {results['f1']:.4f}")
    print(f"\n📝 Logged iteration {iteration}")
    
    # Generate plots every 3 iterations
    if iteration >= 2:
        generate_plots()
    
    print("\n" + "="*60)
    print("COMPLETE - Agent should modify create_features() next")
    print("="*60)


if __name__ == "__main__":
    main()
