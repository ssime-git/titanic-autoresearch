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

    # ITERATION 2: FamilySize
    # Hypothesis: Passengers traveling alone had different survival rates than
    # those with family. Solo travelers may have been more mobile, while families
    # had to coordinate and potentially sacrifice spots. Also large families may
    # have struggled to all get on lifeboats.
    if df_raw is not None:
        X_new["family_size"] = df_raw["sibsp"] + df_raw["parch"] + 1

    # ITERATION 3: Title extraction from Name
    # Hypothesis: Social status (captured in titles like Master, Mrs, Mr)
    # was a significant survival factor. Women and children (Master) were
    # prioritized for lifeboats. Titles capture age+gender+status combination.
    if df_raw is not None and "name" in df_raw.columns:
        # Extract title using regex
        titles = df_raw["name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        # Group rare titles
        title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Dr": "Rare",
            "Rev": "Rare",
            "Col": "Rare",
            "Major": "Rare",
            "Mlle": "Miss",
            "Countess": "Rare",
            "Lady": "Rare",
            "Jonkheer": "Rare",
            "Don": "Rare",
            "Dona": "Rare",
            "Mme": "Mrs",
            "Capt": "Rare",
            "Sir": "Rare",
        }
        titles = titles.map(title_mapping).fillna("Rare")
        # One-hot encode titles
        title_dummies = pd.get_dummies(titles, prefix="title")
        for col in title_dummies.columns:
            X_new[col] = title_dummies[col].astype(float)

    return X_new
