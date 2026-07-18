"""
Data Preprocessing Module
==========================

Handles all data loading, cleaning, and preparation steps for
the Credit Card Fraud Detection project.

Functions:
    - load_dataset: Load the credit card transaction dataset
    - dataset_overview: Display comprehensive dataset summary
    - check_missing_values: Analyze missing values
    - check_duplicates: Detect and handle duplicate records
    - analyze_class_distribution: Analyze fraud vs legitimate distribution
    - verify_data_types: Check and verify column data types
    - split_data: Perform train/test split
    - scale_features: Apply StandardScaler to features
    - preprocess_pipeline: Run complete preprocessing pipeline
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils import get_project_root, set_plot_style

# Logger
logger = logging.getLogger("CreditCardFraudDetection")


def load_dataset(filepath: Path = None) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Args:
        filepath: Path to the CSV file. If None, uses default location.
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if filepath is None:
        filepath = get_project_root() / "dataset" / "creditcard.csv"
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}\n"
            f"Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"Place 'creditcard.csv' in the 'dataset/' folder."
        )
    
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded successfully | Shape: {df.shape}")
    return df


def dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Display comprehensive dataset overview.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dict containing overview statistics
    """
    overview = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "total_missing": df.isnull().sum().sum(),
        "total_duplicates": df.duplicated().sum(),
    }
    
    logger.info(f"Dataset Shape: {overview['shape']}")
    logger.info(f"Total Missing Values: {overview['total_missing']}")
    logger.info(f"Total Duplicates: {overview['total_duplicates']}")
    logger.info(f"Memory Usage: {overview['memory_usage_mb']:.2f} MB")
    
    print("\n--- Dataset Info ---")
    print(f"Rows: {overview['shape'][0]}")
    print(f"Columns: {overview['shape'][1]}")
    print(f"Memory: {overview['memory_usage_mb']:.2f} MB")
    print(f"\n--- First 5 Rows ---")
    print(df.head())
    print(f"\n--- Statistical Summary ---")
    print(df.describe())
    
    return overview


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Missing value summary
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_pct
    }).sort_values(by="Missing_Count", ascending=False)
    
    total_missing = missing.sum()
    logger.info(f"Total missing values: {total_missing}")
    
    if total_missing == 0:
        logger.info("No missing values found in the dataset!")
    else:
        logger.warning(f"Found {total_missing} missing values")
        print(missing_df[missing_df["Missing_Count"] > 0])
    
    return missing_df


def check_duplicates(df: pd.DataFrame, remove: bool = True) -> pd.DataFrame:
    """
    Detect and optionally remove duplicate records.
    
    Args:
        df: Input DataFrame
        remove: Whether to remove duplicates (default: True)
        
    Returns:
        pd.DataFrame: DataFrame with duplicates handled
    """
    duplicate_count = df.duplicated().sum()
    logger.info(f"Duplicate records found: {duplicate_count}")
    
    if duplicate_count > 0 and remove:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Duplicates removed | New shape: {df.shape}")
    
    return df


def analyze_class_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the distribution of fraud vs legitimate transactions.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dict containing class distribution statistics
    """
    class_counts = df["Class"].value_counts()
    class_pcts = df["Class"].value_counts(normalize=True) * 100
    
    distribution = {
        "legitimate_count": int(class_counts.get(0, 0)),
        "fraud_count": int(class_counts.get(1, 0)),
        "legitimate_pct": float(class_pcts.get(0, 0)),
        "fraud_pct": float(class_pcts.get(1, 0)),
        "imbalance_ratio": float(class_counts.get(0, 1) / max(class_counts.get(1, 1), 1)),
    }
    
    logger.info(f"Legitimate Transactions: {distribution['legitimate_count']} ({distribution['legitimate_pct']:.3f}%)")
    logger.info(f"Fraudulent Transactions: {distribution['fraud_count']} ({distribution['fraud_pct']:.3f}%)")
    logger.info(f"Imbalance Ratio: {distribution['imbalance_ratio']:.1f}:1")
    
    print(f"\n--- Class Distribution ---")
    print(f"Legitimate (0): {distribution['legitimate_count']} ({distribution['legitimate_pct']:.3f}%)")
    print(f"Fraud (1):      {distribution['fraud_count']} ({distribution['fraud_pct']:.3f}%)")
    print(f"Imbalance Ratio: {distribution['imbalance_ratio']:.1f}:1")
    
    return distribution


def verify_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check and verify column data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        pd.DataFrame: Data type summary
    """
    dtype_summary = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": df.dtypes.values,
        "Non_Null_Count": df.notnull().sum().values,
        "Null_Count": df.isnull().sum().values,
        "Unique_Values": df.nunique().values
    })
    
    logger.info(f"Data types verified for {len(df.columns)} columns")
    return dtype_summary


def split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        test_size: Proportion for test set
        random_state: Random seed
        stratify: Whether to stratify split by target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Train fraud ratio: {y_train.mean():.4f}")
    logger.info(f"Test fraud ratio: {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns_to_scale: list = None,
    save_scaler: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to specified features.
    
    Args:
        X_train: Training features
        X_test: Testing features
        columns_to_scale: Columns to scale (default: ['Time', 'Amount'])
        save_scaler: Whether to save the scaler
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler)
    """
    if columns_to_scale is None:
        columns_to_scale = ["Time", "Amount"]
    
    scaler = StandardScaler()
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
    
    if save_scaler:
        scaler_path = get_project_root() / "models" / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    logger.info(f"Features scaled: {columns_to_scale}")
    
    return X_train, X_test, scaler


def preprocess_pipeline(
    filepath: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        filepath: Path to dataset CSV
        
    Returns:
        Tuple of (df_clean, X_train, X_test, y_train, y_test, scaler)
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Step 1: Load dataset
    df = load_dataset(filepath)
    
    # Step 2: Dataset overview
    dataset_overview(df)
    
    # Step 3: Check missing values
    check_missing_values(df)
    
    # Step 4: Check duplicates
    df = check_duplicates(df, remove=True)
    
    # Step 5: Analyze class distribution
    analyze_class_distribution(df)
    
    # Step 6: Verify data types
    verify_data_types(df)
    
    # Step 7: Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 8: Scale features
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    logger.info("Preprocessing pipeline completed successfully!")
    
    return df, X_train, X_test, y_train, y_test, scaler
