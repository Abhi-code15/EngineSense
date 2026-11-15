"""
Data preparation script for the predictive maintenance project.

Responsibilities:
- Load the raw engine dataset from the Hugging Face dataset repo (preferred)
  or from the local data folder as a fallback.
- Clean and preprocess the data (rename columns, drop duplicates, basic sanity checks).
- Strictly isolate feature preprocessing: NO median imputation on the whole dataset.
- Reject missing target values.
- Split the cleaned data into train, validation, and holdout test sets (stratified, random_state=42).
- Save train, validation, and test CSVs locally.
- Upload splits to Hugging Face dataset repo if configured with valid credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from hf_data_utils import download_dataset_file, upload_dataset_file


def _load_raw_data_from_hf_or_local() -> pd.DataFrame:
    """
    Try to load the raw dataset from the Hugging Face dataset repo.
    If that fails (e.g., no token or repo yet), fall back to the local CSV.
    """
    if config.HF_TOKEN and config.HF_DATASET_REPO:
        try:
            remote_path = download_dataset_file(
                filename="data/engine_data.csv",
                repo_id=config.HF_DATASET_REPO,
                token=config.HF_TOKEN,
                local_dir=config.DATA_DIR,
            )
            return pd.read_csv(remote_path)
        except Exception as e:
            print(f"Info: Could not load raw data from HF dataset repo ({e}). Using local raw file.")

    if not config.RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {config.RAW_DATA_FILE}. "
            "Ensure engine_data.csv exists or upload it to the HF dataset repo."
        )

    return pd.read_csv(config.RAW_DATA_FILE)


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning and validation without data leakage.

    Key guarantees:
    - Standardize column names.
    - Drop duplicate rows.
    - Explicitly handle missing target values (never impute target as a feature).
    - Validate target binary domain {0, 1}.
    - Feature missing values are NOT imputed here. Imputation is deferred
      to the Scikit-learn Pipeline on training folds to prevent data leakage.
    """
    # Standardize column names
    df = df.rename(columns=config.RAW_COLUMN_RENAME_MAP)

    # Keep only the expected columns (drop any extras, if present)
    expected_cols = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    df = df[[col for col in df.columns if col in expected_cols]]

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Explicitly check and reject missing target values
    if df[config.TARGET_COLUMN].isna().any():
        missing_count = int(df[config.TARGET_COLUMN].isna().sum())
        print(f"Warning: Dropping {missing_count} rows with missing target values ({config.TARGET_COLUMN}).")
        df = df.dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)

    # Validate target values BEFORE converting them to integers.
    # Accept strictly binary values 0 and 1 (including exact floats 0.0, 1.0).
    # Reject fractional values (e.g. 0.5, 1.8), booleans, or other invalid values.
    target_series = df[config.TARGET_COLUMN]
    for val in target_series.unique():
        if isinstance(val, bool):
            raise ValueError(f"Invalid target type: boolean {val!r}. Target must be numeric binary 0 or 1.")
        if isinstance(val, (int, float, np.integer, np.floating)):
            if val not in (0, 1, 0.0, 1.0) or (isinstance(val, (float, np.floating)) and not val.is_integer()):
                raise ValueError(
                    f"Invalid target value detected: {val!r}. Target must be strictly binary 0 or 1 (fractional/out-of-range values not permitted)."
                )
        else:
            raise ValueError(
                f"Invalid target type/value detected: {val!r} of type {type(val).__name__}. Target must be numeric binary 0 or 1."
            )

    # Convert to integer only after strict validation passes
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].astype(int)

    # NOTE: Feature missing values are preserved (not imputed here).
    # Imputation is performed inside the sklearn Pipeline fitted only on training data.
    return df


def _train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataframe into train and test sets (80:20 holdout).
    Maintains backward compatibility with original baseline splits.
    """
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[config.TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test

    return train_df, test_df


def _train_val_test_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split cleaned dataframe into train, validation, and holdout test sets.

    Step 1: 80% train_full, 20% holdout test (stratified).
    Step 2: 80% train_split, 20% validation from train_full (stratified).
    Resulting proportions of total dataset:
    - Train: 64% (12,502 rows)
    - Validation: 16% (3,126 rows)
    - Holdout Test: 20% (3,907 rows)
    """
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]

    # Step 1: Holdout test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Step 2: Stratified train/validation split from train_full
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    train_df = X_train.copy()
    train_df[config.TARGET_COLUMN] = y_train

    val_df = X_val.copy()
    val_df[config.TARGET_COLUMN] = y_val

    test_df = X_test.copy()
    test_df[config.TARGET_COLUMN] = y_test

    return train_df, val_df, test_df


def main() -> None:
    """
    Execute the full leakage-free data preparation pipeline.
    """
    print("Loading raw data...")
    raw_df = _load_raw_data_from_hf_or_local()
    print(f"Raw data shape: {raw_df.shape}")

    print("Cleaning data (without pre-split imputation to prevent leakage)...")
    clean_df = _clean_data(raw_df)
    print(f"Clean data shape: {clean_df.shape}")

    print("Performing stratified 3-way train/validation/test split...")
    train_df, val_df, test_df = _train_val_test_split(clean_df)
    print(
        f"Train shape: {train_df.shape} ({len(train_df)/len(clean_df)*100:.1f}%), "
        f"Val shape: {val_df.shape} ({len(val_df)/len(clean_df)*100:.1f}%), "
        f"Test shape: {test_df.shape} ({len(test_df)/len(clean_df)*100:.1f}%)"
    )

    # Save validation and train-split files locally
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(config.VAL_FILE, index=False)
    train_df.to_csv(config.TRAIN_SPLIT_FILE, index=False)
    print(f"Saved validation set to {config.VAL_FILE}")
    print(f"Saved training split to {config.TRAIN_SPLIT_FILE}")

    # Preserve baseline train.csv and test.csv if they exist
    if not config.TRAIN_FILE.exists():
        train_full, _ = _train_test_split(clean_df)
        train_full.to_csv(config.TRAIN_FILE, index=False)
        print(f"Saved baseline train to {config.TRAIN_FILE}")
    else:
        print(f"Preserved existing baseline train at {config.TRAIN_FILE}")

    if not config.TEST_FILE.exists():
        _, test_full = _train_test_split(clean_df)
        test_full.to_csv(config.TEST_FILE, index=False)
        print(f"Saved baseline test to {config.TEST_FILE}")
    else:
        print(f"Preserved existing baseline test at {config.TEST_FILE}")

    # Upload to HF dataset repo, if configured
    if config.HF_TOKEN and config.HF_DATASET_REPO:
        try:
            print("Uploading dataset splits to Hugging Face dataset repo...")
            upload_dataset_file(
                local_path=config.TRAIN_SPLIT_FILE,
                repo_id=config.HF_DATASET_REPO,
                repo_path="data/train_split.csv",
                token=config.HF_TOKEN,
            )
            upload_dataset_file(
                local_path=config.VAL_FILE,
                repo_id=config.HF_DATASET_REPO,
                repo_path="data/val.csv",
                token=config.HF_TOKEN,
            )
            upload_dataset_file(
                local_path=config.TEST_FILE,
                repo_id=config.HF_DATASET_REPO,
                repo_path="data/test.csv",
                token=config.HF_TOKEN,
            )
            print("Upload to Hugging Face completed successfully.")
        except Exception as e:
            print(f"ERROR: Failed to upload splits to Hugging Face dataset repo: {e}")
    else:
        print("Info: HF_TOKEN not configured. Skipping Hugging Face dataset upload. All files saved locally.")


if __name__ == "__main__":
    main()
