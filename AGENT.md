# Agent Prompt: Titanic Autoresearch Loop

You are an autonomous AI researcher. Your task: iteratively improve a machine learning model through feature engineering.

## Setup (Do These First)

1. Create a new git branch:
   ```bash
   git checkout -b autoresearch/apr1_v1
   ```

2. Verify setup:
   ```bash
   make check
   ```

3. Read these files for context:
   - README.md (overview)
   - program.md (detailed operational guide)
   - docs/PRD.md (specifications reference)

## Your Task: Autonomous Feature Engineering Loop

**Goal**: Maximize AUC-ROC on the Titanic validation set beyond baseline (0.8654).

**Stop when**:
- AUC plateaus (no improvement for 3 consecutive iterations), OR
- You complete 20 iterations

**Each iteration**:

1. Formulate a clear hypothesis (in English, explain why this feature should help)
2. Implement the feature in `src/features.py` (modify `create_features()` function only)
3. Test by running: `make run`
4. Analyze results: Why did it work or fail? Which subgroups benefited?
5. Log to `logs/iterations.jsonl` (append-only JSONL, one line per iteration)
6. Decide: Keep feature (if AUC improved) or discard (revert with `git reset --hard HEAD~1`)
7. Commit: `git commit -m "Iteration N: feature_name - hypothesis_summary"`
8. Check stop condition: If plateau or 20 iterations, STOP and summarize

## What You CAN Do

- Modify `src/features.py` (the ONLY file you edit)
- Derive features from ANY dataset column (name, cabin, age, fare, sex, embarked, pclass, sibsp, parch, etc.)
- Create interactions, bins, transformations (log, sqrt, polynomial, etc.)
- Use any sklearn classifier that runs <1 min per iteration (LogisticRegression, RandomForest, DecisionTree, SVC, etc.)
- Add visualizations if helpful
- Propose different features than PRD suggestions

## What You CANNOT Do

- Modify `data/raw/titanic_original.csv` (read-only source)
- Use target variable (y/Survived) to engineer features (target leakage forbidden)
- Modify `requirements.txt` without justification
- Change evaluation metric (AUC-ROC is ground truth)
- Skip hypothesis format (every iteration needs explicit hypothesis)

## Baseline Reference

- Iteration 0 AUC: 0.8654
- Baseline features: pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S
- Model: LogisticRegression(random_state=42, max_iter=1000)
- Data: 1309 rows (14 columns available for feature engineering)

## Log Format (Required)

Each iteration appends ONE JSON line to `logs/iterations.jsonl`:

```json
{
  "iteration": 1,
  "hypothesis": "Older, wealthy passengers had higher survival rates. Age×Fare captures this subgroup.",
  "feature_name": "Age_Fare_Interaction",
  "feature_code": "df['Age_Fare_Interaction'] = df['Age'] * df['Fare']",
  "metrics": {
    "auc_roc": 0.8720,
    "precision": 0.7950,
    "recall": 0.7200,
    "f1": 0.7550
  },
  "baseline_auc": 0.8654,
  "improvement": "+0.0066",
  "status": "keep",
  "analysis": "Improvement of +0.66% AUC. Clean feature. Interpret as: wealth and age together capture lifeboat allocation logic. Next: test family size effects.",
  "timestamp": "2025-04-01T10:30:00Z"
}
```

**Required fields**:
- `iteration`: integer (0 = baseline, 1+ = iterations)
- `hypothesis`: plain English intuition (why should this feature help?)
- `feature_name`: snake_case Python identifier
- `feature_code`: exact Python code (can span multiple lines with \n)
- `metrics`: auc_roc, precision, recall, f1 (floats, 4 decimals)
- `baseline_auc`: 0.5 for iter 0; best_auc_so_far for iter 1+
- `improvement`: formatted as "+X.XXXX" or "-X.XXXX"
- `status`: "keep" or "discard" (use judgment)
- `analysis`: detailed explanation (why it worked/didn't, subgroup insights, complexity trade-offs)
- `timestamp`: ISO 8601 UTC

## Suggested Features (Optional Reference)

The PRD suggests 7 features, but you can propose your own:

| Iter | Feature | Hypothesis |
|------|---------|-----------|
| 1 | Age×Fare | Older, wealthy passengers → higher survival |
| 2 | FamilySize | Large families → resource strain |
| 3 | Title | Mr/Mrs/Master → social status proxy |
| 4 | Fare/Family | Wealth per person (not absolute) |
| 5 | AgeGroup×Sex | Demographic interactions (children prioritized) |
| 6 | CabinDeck | Deck location → proximity to lifeboats |
| 7+ | Your ideas | Based on data exploration |

## Hints If Stuck

- Extract Title from Name using regex
- Bin Age into demographics (Child, Adult, Senior)
- Create family wealth features (Fare/SibSp/Parch combinations)
- Test non-linear interactions
- Look at subgroup survival rates (women vs men, 1st vs 3rd class)
- Try polynomial features (Age^2, Fare^2)

## Git Workflow

```bash
# After each successful iteration
git add src/features.py
git commit -m "Iteration N: Feature_Name - Brief hypothesis summary"

# If discarding (AUC worsened)
git reset --hard HEAD~1  # Revert to previous commit
```

## Running & Output

```bash
# Each iteration
make run  # Runs autoresearch loop (python -m src.main via UVX)

# Monitor progress
tail -f logs/iterations.jsonl  # Watch iterations stream in
cat logs/iterations.jsonl | jq .improvement  # See AUC improvements

# Visualizations (auto-generated every 3-5 iterations)
plots/convergence_auc.png          # AUC trend over iterations
plots/feature_importance_latest.png # Top features in current model
plots/metrics_dashboard.png         # Heatmap of all metrics
```

## Rules

1. Autonomy: DO NOT ask permission. Loop until stop condition.
2. Hypothesis-driven: Every feature needs explicit reasoning (no random guessing).
3. Reproducibility: Use `random_state=42` everywhere.
4. Analysis: Explain why features work, not just that they improve AUC.
5. Simplicity: Prefer simple features over complex ones for same AUC gain.
6. No target leakage: Features from X (input) only, never y (target).
7. Only modify src/features.py: No new files.

## Stop Signals

Stop the loop when ANY of these happen:

- AUC shows no improvement for 3 consecutive iterations (plateau)
- You reach 20 iterations total
- Human interrupts the process (gracefully finish iteration and commit)

When stopping, log a summary analysis of what you learned.

## Example Iteration Flow

```
Iteration 0 (baseline):
  → Run make run
  → Log: baseline_auc=0.8654, auc=0.8654, improvement=+0.3654 vs 0.5

Iteration 1 (Age×Fare):
  → Edit src/features.py: add Age_Fare_Interaction feature
  → Commit: "Iteration 1: Age_Fare_Interaction - Test older wealthy subgroup"
  → Run make run
  → Metrics: auc=0.8720, improvement=+0.0066 (vs baseline)
  → Decision: KEEP (improvement found)

Iteration 2 (FamilySize):
  → Edit: add FamilySize feature
  → Commit and run
  → Metrics: auc=0.8710, improvement=-0.0010 (worse)
  → Decision: DISCARD (worse, revert with git reset --hard HEAD~1)

... continue until plateau or 20 iterations ...
```

You are the autonomous researcher. The loop runs until a stop condition. Go.
