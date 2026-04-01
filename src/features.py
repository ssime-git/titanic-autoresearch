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

    # ITERATION 0: Baseline - no features added
    # TODO: Agent adds feature engineering code here in next iteration

    return X_new
