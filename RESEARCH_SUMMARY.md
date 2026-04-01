# Titanic Autoresearch Summary

Branch: autoresearch/apr1_v1
Date: 2026-04-01

## Goal
Iteratively improve a Titanic survival prediction model through feature engineering,
maximizing AUC-ROC beyond baseline (0.8654).

Stop conditions:
- AUC plateaus (no improvement for 3 consecutive iterations)
- Complete 20 iterations maximum

---

## Methodology

Each iteration followed this process:
1. Formulate hypothesis (why this feature should help)
2. Implement feature in src/features.py
3. Run make run to evaluate
4. Analyze results and decide keep/discard
5. Commit if kept, revert if discarded
6. Check stop condition

---

## Iteration Results

### Iteration 0: Baseline
- AUC: 0.8654
- Features: pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S
- Status: Baseline reference

### Iteration 1: Age x Fare Interaction
- Hypothesis: Older, wealthier passengers had higher survival due to social status
- AUC: 0.8641
- Improvement: -0.0013
- Status: DISCARDED
- Analysis: No predictive value in the interaction

### Iteration 2: FamilySize
- Hypothesis: Solo travelers vs families had different survival patterns
- Code: sibsp + parch + 1
- AUC: 0.8655
- Improvement: +0.0001
- Status: KEPT
- Analysis: Foundation feature, minimal but positive improvement

### Iteration 3: Title Extraction
- Hypothesis: Social status (Master, Mrs, Mr, Miss) captured evacuation priority
- Code: Extracted title from Name using regex, one-hot encoded
- AUC: 0.8820
- Improvement: +0.0165
- Status: KEPT
- Analysis: Major breakthrough. Title captures age+gender+status combination

### Iteration 4: IsAlone
- Hypothesis: Solo travelers may have had survival advantage
- Code: FamilySize == 1
- AUC: 0.8818
- Improvement: -0.0002
- Status: DISCARDED
- Analysis: Redundant with FamilySize

### Iteration 5: FarePerPerson
- Hypothesis: Per-capita wealth more meaningful than absolute fare
- Code: Fare / FamilySize
- AUC: 0.8816
- Improvement: -0.0004
- Status: DISCARDED
- Analysis: Absolute fare already sufficient

### Iteration 6: Deck Extraction
- Hypothesis: Cabin deck level determined proximity to lifeboats
- Code: Extracted first letter from Cabin
- AUC: 0.8813
- Improvement: -0.0007
- Status: DISCARDED
- Analysis: Too many missing cabin values, no predictive power

### Iteration 7: AgeGroup Bins
- Hypothesis: Non-linear age effects (child/adult/senior)
- Code: Binned age into 0-16, 16-50, 50+ groups
- AUC: 0.8815
- Improvement: -0.0005
- Status: DISCARDED
- Analysis: Continuous age already captures the pattern

---

## Stop Condition

Stopped after 4 consecutive iterations without improvement (Iters 4-7).
Plateau detected at AUC 0.8820.

---

## Final Results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8820 |
| Baseline | 0.8654 |
| Improvement | +0.0166 (+1.92%) |
| Precision | 0.8085 |
| Recall | 0.7600 |
| F1 | 0.7835 |

---

## Features in Final Model

1. family_size: SibSp + Parch + 1
2. title_Master: Boys (strong positive coefficient)
3. title_Mrs: Married women (positive coefficient)
4. title_Miss: Unmarried women (slight positive)
5. title_Mr: Men (strong negative coefficient)
6. title_Rare: Rare titles (slight negative)

Plus baseline features: pclass, age, sibsp, parch, fare, sex_male, embarked_Q, embarked_S

---

## Key Insights

1. Title extraction was the breakthrough feature (+1.65% AUC)
2. Titles capture the "women and children first" evacuation policy
3. Master (boys) had highest importance - they were prioritized with women
4. Mr (men) had strongly negative coefficient - lowest survival priority
5. Family structure matters but binary indicators (IsAlone) add no value
6. Wealth per capita not better than absolute fare
7. Deck information too sparse to be useful
8. Age binning not better than continuous age

---

## Files Generated

- logs/iterations.jsonl: Complete experiment log with all metrics
- plots/convergence_auc.png: AUC progression over iterations
- plots/feature_importance_latest.png: Feature coefficients from final model
- plots/metrics_dashboard.png: Heatmap of all metrics by iteration

---

## Commands Used

```bash
# Setup
git checkout -b autoresearch/apr1_v1
make check

# Run iterations
make run

# Code quality
make lint

# Git workflow (per iteration)
git add src/features.py
git commit -m "Iteration N: description"

# Discard if needed
git reset --hard HEAD~1
```

---

## Conclusion

Successfully improved Titanic model AUC from 0.8654 to 0.8820 (+1.92%)
using hypothesis-driven feature engineering. Title extraction from Name
provided the largest gain by encoding social status and evacuation priority.
Further iterations hit plateau, indicating feature engineering saturation
for this model architecture.
