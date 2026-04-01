"""Data loading and preprocessing module."""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Load raw Titanic data with all columns available.

    Agent can derive features from any column in the dataset.

    Args:
        data_path: Path to the Titanic CSV file.

    Returns:
        DataFrame with raw Titanic data.

    Raises:
        FileNotFoundError: If data file does not exist.
        pd.errors.ParserError: If CSV is malformed.
    """
    try:
        logger.info(f"Loading raw data from {data_path}...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        df.columns = df.columns.str.lower()

        initial_rows = len(df)
        df = df.dropna(subset=["survived"])
        removed_rows = initial_rows - len(df)
        logger.info(
            f"Removed {removed_rows} rows with missing target: {len(df)} rows remain"
        )
        logger.info(f"Available columns: {list(df.columns)}")

        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found at {data_path}")
        raise e
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file: {e}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise e


def preprocess_baseline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply baseline preprocessing: imputation and one-hot encoding.

    Handles missing values and categorical encoding to create X, y ready
    for train_test_split.

    Args:
        df: Raw input DataFrame with all columns.

    Returns:
        Tuple of (X, y) where X is numeric features and y is target.

    Raises:
        KeyError: If expected columns are missing.
        ValueError: If preprocessing fails.
    """
    try:
        df = df.drop(
            columns=["passengerid", "ticket", "boat", "body", "home.dest"],
            errors="ignore",
        )

        y = df["survived"].copy()
        df = df.drop(columns=["survived"])

        df["age"] = df.groupby("pclass")["age"].transform(
            lambda x: x.fillna(x.median())
        )

        df["fare"] = df["fare"].fillna(df["fare"].median())

        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

        df = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=True)

        X = df.select_dtypes(include=[np.number, bool]).astype(float)

        return X, y
    except KeyError as e:
        logger.error(f"Missing expected column: {e}")
        raise e
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise e


def prepare_data_for_training(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data and apply standardization.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Test set fraction.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train, y_test).

    Raises:
        ValueError: If data shapes are invalid.
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        return X_tr_s, X_te_s, y_tr, y_te
    except ValueError as e:
        logger.error(f"Data preparation failed: {e}")
        raise e
