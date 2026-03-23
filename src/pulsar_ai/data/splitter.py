"""Dataset splitting utilities for train/validation/test sets."""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def train_test_split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.1,
    seed: Optional[int] = None,
    stratify_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets.

    Args:
        df: Source DataFrame.
        test_size: Fraction of data for the test set (0.0–1.0).
        seed: Random seed for reproducibility.
        stratify_column: Column name for stratified splitting.

    Returns:
        Tuple of (train_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split dataset: %d train, %d test (test_size=%.2f)",
        len(train_df),
        len(test_df),
        test_size,
    )
    return train_df, test_df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.1,
    seed: Optional[int] = None,
    stratify_column: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into train and test sets (dict interface).

    Wrapper around train_test_split_dataset that returns a dict
    with 'train' and 'test' keys as expected by the SFT trainer.

    Args:
        df: Source DataFrame.
        test_size: Fraction of data for the test set.
        seed: Random seed.
        stratify_column: Column for stratified splitting.

    Returns:
        Dict with 'train' and 'test' DataFrames.
    """
    train_df, test_df = train_test_split_dataset(
        df, test_size=test_size, seed=seed, stratify_column=stratify_column,
    )
    return {"train": train_df, "test": test_df}
