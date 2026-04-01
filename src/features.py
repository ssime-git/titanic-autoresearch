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

    return X_new
