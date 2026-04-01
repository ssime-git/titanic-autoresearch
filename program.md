# titanic-autoresearch

Experiment to have an LLM autonomously iterate on feature engineering hypotheses using the autoresearch loop pattern.

## Setup

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr1_v1`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   * `README.md` — repository overview and quick start.
   * `src/autoresearch_loop.py` — the main experiment file. This is the ONLY file you modify.
   * `docs/PRD.md` — full specifications (reference, not a must-read every time).
4. **Prepare data**: Check that `data/raw/titanic_original.csv` exists. If not, download from [Geoyi/Cleaning-Titanic-Data](https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv).
5. **Initialize logs/iterations.jsonl**: Create `logs/iterations.jsonl` as an empty file. The first hypothesis will create the first log entry.
6. **Verify requirements**: Using UVX: `uvx --with pandas --with scikit-learn --with matplotlib --with seaborn python notebooks/autoresearch_loop.py`
7. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Experimentation

Each iteration runs a single feature engineering hypothesis. Agent stops when:
- **AUC plateaus** (no improvement for 3 consecutive iterations), OR
- **20 iterations completed**, whichever comes first.

**What you CAN do:**

* Modify `src/autoresearch_loop.py` — this is the only file you edit.
* **Preprocess flexibly**: Extract features from raw columns (e.g., Title from Name), create derived features (FamilySize = SibSp + Parch + 1), bin continuous features, etc.
* Create new feature engineering functions within the file.
* Include more raw data columns if needed (e.g., Name for Title extraction, or any other columns). Only constraint: all preprocessing must happen AFTER train/test split to avoid data leakage.
* Change model class: Use any sklearn classifier (LogisticRegression, RandomForest, DecisionTree, SVC, GradientBoosting, etc.) as long as a single iteration completes in <1 minute.
* Adjust model hyperparameters (C for LR, max_depth for RF, etc.) if they help.
* Add new visualizations if they provide insight.
* Propose different features than the suggestions in the PRD.

**What you CANNOT do:**

* Modify `data/raw/titanic_original.csv`. It is read-only. This is the source truth.
* Modify `requirements.txt` without explicit reason. New packages should only be added if absolutely justified.
* Change the evaluation metric. `AUC-ROC` is the ground truth metric. Also report: Precision, Recall, F1.
* Modify `data/processed/` files manually — they are generated outputs.
* Skip the structured hypothesis format. Every iteration MUST have an explicit hypothesis in plain English.
* **Use target variable (y/Survived) to engineer features**. Features MUST derive ONLY from input features (X). Examples of forbidden:
  - Computing survival rate by port and using it as a feature (target leakage)
  - Using any y-based statistic in feature engineering
  - Only allowed: Domain knowledge and input feature transformations

**The goal is simple: maximize AUC-ROC on the validation set.** Each iteration should:

1. Formulate a **clear hypothesis** (in English, not code)
2. Implement the feature engineering based on that hypothesis
3. Train a model (any sklearn classifier, runs <1 min)
4. Evaluate and measure metrics (AUC, Precision, Recall, F1)
5. Analyze the results: *why did it work or not? What subgroup benefited?*
6. Log everything to `logs/iterations.jsonl`
7. Decide: keep this feature or discard?
8. Move to the next hypothesis based on the analysis

**Simplicity criterion**: All else being equal, simpler features win. A feature that adds 3 lines of code and gains +0.02 AUC is better than a feature that requires 30 lines and gains +0.02 AUC. If you can simplify an existing feature without losing AUC, that's a win — do it.

**The first run**: Always establish the baseline first. Run with only the preprocessing, no feature engineering. Log this as "Iteration 0" with status "baseline". This is your reference point.

## Baseline Reference

- **Iteration 0 AUC**: 0.8654 (achieved without feature engineering)
- **Features in baseline**: pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S
- **Rows after preprocessing**: 1309 (dropped 1 row with missing Survived)
- **Model for baseline**: LogisticRegression(random_state=42, max_iter=1000)

Use this as the reference point. Iteration 1+ should try to beat 0.8654.

## Output format

Each iteration writes a single JSON object to `logs/iterations.jsonl` (one per line — JSONL format, NOT one big JSON object).

Structure:

```json
{
  "iteration": 1,
  "hypothesis": "Passengers who were both older and paid higher fares were more likely to survive. The interaction of age and wealth captures a distinct subgroup.",
  "feature_name": "Age_Fare_Interaction",
  "feature_code": "df['Age_Fare_Interaction'] = df['Age'] * df['Fare']",
  "metrics": {
    "auc_roc": 0.812,
    "precision": 0.78,
    "recall": 0.65,
    "f1": 0.71
  },
  "baseline_auc": 0.8654,
  "improvement": "+0.0466",
  "status": "keep",
  "analysis": "Interaction captures the subgroup of older, wealthy passengers. Strong improvement (+4.7% AUC). Feature is clean and interpretable. Worth keeping. Next: test Family Size effects to see if group dynamics matter.",
  "timestamp": "2025-04-01T10:23:45Z"
}
```

**Fields (required)**:
- `iteration`: integer, starting from 0 (baseline)
- `hypothesis`: plain English description of the intuition — why should this feature help?
- `feature_name`: short name, used in code (e.g. `Age_Fare_Interaction`, must be snake_case)
- `feature_code`: the actual Python line(s) to create the feature (can include \n for multiline)
- `metrics`: dict with floats (4 decimal places): `auc_roc`, `precision`, `recall`, `f1`
- `baseline_auc`: for iteration 0, use 0.5 (reference). For iteration 1+, use best_auc_so_far (the highest AUC from all previous iterations).
- `improvement`: delta as "+X.XXXX" or "-X.XXXX" compared to baseline_auc
- `status`: `keep` or `discard` — your decision. Use judgment: weigh improvement vs complexity.
- `analysis`: explain why it worked or didn't. Be specific: which subgroups benefited? Where did it fail? Is the complexity worth the gain?
- `timestamp`: ISO format (UTC)

**DO NOT manually edit `logs/iterations.jsonl`**. Only append new lines programmatically at the end of each run.

## Logging to iterations.jsonl

After each run completes, append exactly one line to `logs/iterations.jsonl`. Use Python's `json.dumps()` and write in append mode:

```python
import json
from datetime import datetime

result = {
    "iteration": 1,
    "hypothesis": "...",
    # ... all fields from above
    "timestamp": datetime.utcnow().isoformat() + "Z"
}

with open("logs/iterations.jsonl", "a") as f:
    f.write(json.dumps(result) + "\n")
```

Ensure the JSON is valid (use `json.dumps`, not manual string concatenation).

## Visualization generation

After every 3-5 iterations, generate three plots in `plots/`:

### Plot 1: convergence_auc.png
- **Type**: Line plot (matplotlib)
- **X-axis**: Iteration number (0, 1, 2, ...)
- **Y-axis**: AUC-ROC value (range ~0.70 to 0.90)
- **Lines**:
  - Baseline AUC (0.8654) as a horizontal dashed gray line (reference)
  - Actual AUC for each iteration as a blue line with markers
- **Labels**: X: "Iteration", Y: "AUC-ROC", Title: "Feature Engineering Convergence"

### Plot 2: feature_importance_latest.png
- **Type**: Horizontal bar plot (matplotlib)
- **X-axis**: Importance score (0.0 to 1.0 or coefficient magnitude)
- **Y-axis**: Feature names (e.g., Age, Fare, Sex_male, FamilySize, Title_Mr, ...)
- **Colors**: Gradient (high importance = dark, low = light)
- **Source**: Extract from latest fitted model using coefficients (LogisticRegression) or feature_importances (tree-based)

### Plot 3: metrics_dashboard.png
- **Type**: Heatmap (seaborn)
- **Rows**: Iteration numbers (0 to N)
- **Columns**: Metrics (AUC, Precision, Recall, F1)
- **Values**: Metric values from iterations.jsonl
- **Colormap**: Blues (or diverging colormap where 0.5 is neutral)
- **Colorbar**: Show scale (e.g., 0.60 to 0.90)
- **Annot**: True (show values in cells)

Generate these **after every 3-5 iterations**, not after each run (to keep the loop moving fast).

## The experiment loop

You are running on the branch `autoresearch/<tag>`.

**STOP CONDITION:**

Stop the loop when **either**:
1. AUC does NOT improve for 3 consecutive iterations (plateau detected), OR
2. You complete 20 iterations.

Whichever comes first. Log this decision in your final analysis.

**LOOP UNTIL STOP:**

1. Read `logs/iterations.jsonl` to understand what you've tried so far. Count total iterations completed.
2. Analyze the latest results: Is AUC improving? Are features becoming redundant? What insight from the last analysis?
3. Formulate a new hypothesis. Be specific — don't just say "try a new feature". Explain the intuition clearly.
4. Implement the feature in `notebooks/autoresearch_loop.py`. Keep the code clean.
5. git commit with message: `"Iteration N: <feature_name> - <hypothesis_summary>"` (e.g. `"Iteration 3: FamilySize - Test family size effect on survival rates"`)
6. Run the script using UVX: `uvx --with pandas --with scikit-learn --with matplotlib --with seaborn --with numpy python notebooks/autoresearch_loop.py`
7. Wait for completion (should be seconds to ~1 minute per iteration).
8. Check for crashes: If you see Python exceptions, read the full traceback, fix the bug, and re-run. Do NOT skip crashes without attempting a fix.
9. Extract the metrics from logs or script output.
10. Write the results to `logs/iterations.jsonl` by appending a new JSON line.
11. Decide: `status = "keep"` or `discard`. Use judgment:
    - `keep`: if AUC improved, OR if AUC stayed nearly flat but the feature is foundational for next iteration
    - `discard`: if AUC worsened AND the feature adds complexity
12. If you `keep`, leave the feature in the code and proceed. If you `discard`, remove the feature from `autoresearch_loop.py` and do `git reset --hard HEAD~1` to revert the commit.
13. **Check stop condition**: If AUC didn't improve for 3 iterations OR you've done 20 iterations, **STOP**. Otherwise, move to the next hypothesis.

**STOP SIGNALS:**
- Iteration count reaches 20
- Last 3 iterations all had `status: "discard"` (AUC plateau)

When you hit a stop signal, log a final summary message and let the human know you've stopped.

If you run out of immediate ideas before hitting stop condition:
- Re-read the dataset columns and distributions
- Try interactions between features you've already created
- Try non-linear transformations (log, sqrt, polynomial bins, squared terms)
- Look for patterns in subgroups (e.g., features that help women but not men, or 3rd class but not 1st class)
- Try binning continuous features (Age into Child/Adult/Senior, Fare into quartiles)
- Combine previous near-misses ("that didn't work alone, but maybe with X it helps")

**Expected iteration pace**: 1-2 minutes per iteration (data load + preprocessing + train + eval). You can run 20 iterations per hour easily.

**Timeout**: If a run exceeds 5 minutes (excluding logs write time), it's too slow — optimize or simplify. If it exceeds 10 minutes, kill it (`Ctrl+C`) and mark as `crash`.

**Crashes**: If a run crashes (OOM, syntax error, missing data, etc.):
- If it's a typo or simple bug: fix immediately and re-run.
- If it's a fundamental issue with the feature idea: log it as `status: "crash"` and move on. Don't get stuck.

## Git workflow

After each successful run:
```bash
git add notebooks/autoresearch_loop.py
git commit -m "Iteration N: <feature> - <reason>"
```

After a discard:
```bash
git reset --hard HEAD~1  # Revert to last commit
```

**Never commit these files to git:**
- `logs/iterations.jsonl` — this is data, not code. Keep it untracked in `.gitignore`.
- `plots/*.png` — these are outputs, not source. Keep untracked.
- `data/processed/` — generated outputs. Keep untracked.

Only `autoresearch/<tag>` branch commits are code changes (`autoresearch_loop.py`).

## Stopping and resuming

If the experiment is interrupted:
- All state is preserved in `logs/iterations.jsonl` (append-only log) and the git branch history.
- To resume: check out the branch, read the log file to see what's been tried, and continue from the next iteration.

Example resume:
```bash
git checkout autoresearch/apr1_v1
uvx --with pandas --with scikit-learn --with matplotlib --with seaborn --with numpy python notebooks/autoresearch_loop.py
```

## Key Rules

1. **Autonomy**: Do NOT ask the human for permission to continue. Loop until stop condition is hit.
2. **Hypothesis-driven**: Every feature must have an explicit hypothesis. Random features are not acceptable.
3. **Reproducibility**: Always use `random_state=42`. Results must be deterministic.
4. **Analysis**: Write thoughtful analysis. "It improved AUC" is not enough. Explain *why*.
5. **Simplicity**: Prefer simple features to complex ones, all else being equal.
6. **Logging**: Every iteration must be logged to `iterations.jsonl`. No skipped runs.
7. **Only modify autoresearch_loop.py**: All code changes go into this one file. Do not create new files unless unavoidable.
8. **No target leakage**: Features must derive ONLY from input features (X), never from target (y).
9. **Feature names**: Must be valid Python identifiers (snake_case, no spaces or special chars).
10. **Preprocessing happens post-split**: One-hot encoding, imputation, binning — all happen AFTER train/test split to avoid data leakage.

---

**You are the autonomous researcher. The loop runs until a stop condition is hit. Go.** 🚀
