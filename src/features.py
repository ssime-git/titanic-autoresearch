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

    return X_new
