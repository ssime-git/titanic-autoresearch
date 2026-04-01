"""Feature engineering module - AGENT MODIFIES THIS FILE."""

from typing import Optional

import pandas as pd


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

    # AGENT: Add feature engineering code below this line

    # ITERATION 1: Age × Fare Interaction
    # Hypothesis: Older, wealthier passengers (high Age × high Fare) had higher survival
    # rates. This interaction captures "established wealthy" passengers who had
    # priority access to lifeboats due to social status and cabin location.
    # Result: DISCARDED - AUC decreased from 0.8654 to 0.8641 (-0.0013)
    # Analysis: Wealth and age likely act as independent predictors,
    # their interaction adds noise rather than signal.

    # ITERATION 2: FamilySize
    # Hypothesis: Passengers traveling alone had different survival rates than
    # those with family. Solo travelers may have been more mobile during evacuation,
    # while large families faced coordination challenges. Medium families (2-4 members)
    # may have helped each other, while very large families struggled.
    if df_raw is not None:
        X_new["family_size"] = df_raw["sibsp"] + df_raw["parch"] + 1

    # ITERATION 3: Title Extraction from Name
    # Hypothesis: Social titles (Mr, Mrs, Master, Miss) capture age, gender, and status.
    # Women and children (Master title means boy) were prioritized for lifeboats.
    # Titles provide richer signal than sex/age separately.
    if df_raw is not None and "name" in df_raw.columns:
        titles = df_raw["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_mapping = {
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
            "Mlle": "Miss", "Countess": "Rare", "Lady": "Rare",
            "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare",
            "Mme": "Mrs", "Capt": "Rare", "Sir": "Rare",
        }
        titles = titles.map(title_mapping).fillna("Rare")
        title_dummies = pd.get_dummies(titles, prefix="title")
        for col in title_dummies.columns:
            X_new[col] = title_dummies[col].astype(float)

    # ITERATION 4: IsAlone flag
    # Hypothesis: Passengers traveling alone (family size == 1) may have been more mobile and thus higher survival probability.
    if df_raw is not None:
        family_sz = df_raw["sibsp"] + df_raw["parch"] + 1
        X_new["is_alone"] = (family_sz == 1).astype(int)

    # ITERATION 5: Fare per Family (Fare divided by family size)
    # Hypothesis: Wealth per person may be a more relevant predictor than total fare.
    # For passengers traveling alone, this equals fare; for families, it reflects
    # shared expense. May capture economic status more accurately.
    if df_raw is not None:
        family_sz = df_raw["sibsp"] + df_raw["parch"] + 1
        # Avoid division by zero (should not happen as family_sz >=1)
        X_new["fare_per_family"] = X_new["fare"] / family_sz

    # ITERATION 6: Age Group Bins
    # Hypothesis: Survival varies by age categories (children prioritized, seniors less likely). Binning age may capture non-linear effects.
    if X_new.get('age') is not None:
        age_bins = pd.cut(X_new['age'], bins=[0, 12, 18, 40, 60, 100],
                         labels=['child', 'teen', 'adult', 'senior', 'elder'], right=False)
        age_dummies = pd.get_dummies(age_bins, prefix='age_group')
        for col in age_dummies.columns:
            X_new[col] = age_dummies[col].astype(float)

    return X_new
