"""
Data preprocessing module for Titanic Survival Prediction.
Handles missing values, encoding, scaling, and data cleaning.

Author: CodSoft Data Science Intern
Task: Task 1 - Titanic Survival Prediction
"""

from typing import Tuple, Optional
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Get logger from utils
logger = logging.getLogger("TitanicPrediction")


# ============================================================================
# MISSING VALUE HANDLING
# ============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle all missing values in the dataset.
    
    Strategy:
        - Age: Median imputation grouped by Pclass and Sex
        - Embarked: Mode imputation (most common: Southampton)
        - Cabin: Mark as 'Unknown' (too many missing for reliable imputation)
        - Fare: Median imputation grouped by Pclass (if any missing)
    
    Args:
        df: Input DataFrame with missing values
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Track original missing counts
    original_missing = df.isnull().sum()
    
    # Age: Median imputation grouped by Pclass & Sex
    # Rationale: Age distribution varies significantly by class and gender
    if df["Age"].isnull().any():
        df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
        # Fallback for edge cases (e.g., no data in a group)
        df["Age"].fillna(df["Age"].median(), inplace=True)
        logger.info(f"Age: Imputed {original_missing['Age']} missing values (grouped median)")
    
    # Embarked: Mode imputation
    # Rationale: Only 2 missing values; Southampton is most common port
    if df["Embarked"].isnull().any():
        mode_value = df["Embarked"].mode()[0]
        df["Embarked"].fillna(mode_value, inplace=True)
        logger.info(f"Embarked: Imputed {original_missing['Embarked']} missing values (mode: {mode_value})")
    
    # Fare: Median imputation by Pclass
    # Rationale: Fare strongly correlates with passenger class
    if df["Fare"].isnull().any():
        df["Fare"] = df.groupby("Pclass")["Fare"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Fare"].fillna(df["Fare"].median(), inplace=True)
        logger.info(f"Fare: Imputed {original_missing['Fare']} missing values (grouped median)")
    
    # Cabin: Fill with 'Unknown'
    # Rationale: 77% missing - too many to impute reliably; will create binary flag instead
    if df["Cabin"].isnull().any():
        df["Cabin"].fillna("Unknown", inplace=True)
        logger.info(f"Cabin: Marked {original_missing['Cabin']} missing values as 'Unknown'")
    
    return df


# ============================================================================
# DUPLICATE HANDLING
# ============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    else:
        logger.info("No duplicate rows found")
    
    return df


# ============================================================================
# CATEGORICAL ENCODING
# ============================================================================

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables for model training.
    
    Encoding Strategy:
        - Sex: Label Encoding (binary: male=1, female=0)
        - Embarked: One-Hot Encoding (3 categories, no ordinal relationship)
        - Title: Label Encoding (if exists; ordinal-like groups)
    
    Args:
        df: DataFrame with categorical columns
    
    Returns:
        DataFrame with encoded categorical columns
    """
    df = df.copy()
    
    # Label Encode Sex (binary)
    if "Sex" in df.columns and df["Sex"].dtype == "object":
        le_sex = LabelEncoder()
        df["Sex"] = le_sex.fit_transform(df["Sex"])
        logger.info(f"Sex: Label encoded (female=0, male=1)")
    
    # One-Hot Encode Embarked (3 categories, no ordinal relationship)
    if "Embarked" in df.columns and df["Embarked"].dtype == "object":
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int)
        df = pd.concat([df, embarked_dummies], axis=1)
        df.drop("Embarked", axis=1, inplace=True)
        logger.info(f"Embarked: One-hot encoded into {list(embarked_dummies.columns)}")
    
    # Label Encode Title (if exists)
    if "Title" in df.columns and df["Title"].dtype == "object":
        le_title = LabelEncoder()
        df["Title"] = le_title.fit_transform(df["Title"])
        logger.info("Title: Label encoded")
    
    return df


# ============================================================================
# COLUMN MANAGEMENT
# ============================================================================

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns not useful for modeling.
    
    Dropped Columns:
        - PassengerId: Just an identifier, no predictive value
        - Name: Unstructured text (Title already extracted)
        - Ticket: High cardinality, no clear pattern
        - Cabin: High missing rate (CabinAvailable already extracted)
    
    Args:
        df: DataFrame with all columns
    
    Returns:
        DataFrame with unnecessary columns dropped
    """
    df = df.copy()
    
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info(f"Dropped columns: {existing_cols}")
    
    return df


# ============================================================================
# FEATURE SCALING
# ============================================================================

def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to normalize numerical features.
    
    Standardization: (x - mean) / std
    
    Args:
        X: Feature DataFrame to scale
    
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    scaler = StandardScaler()
    
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    logger.info("Features scaled with StandardScaler")
    return X_scaled, scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to new data.
    
    Args:
        X: Feature DataFrame to scale
        scaler: Pre-fitted StandardScaler
    
    Returns:
        Scaled DataFrame
    """
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def full_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.
    
    Pipeline Steps:
        1. Remove duplicates
        2. Handle missing values
    
    Note: This should be called BEFORE feature engineering.
          Encoding and column dropping happen AFTER feature engineering.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    logger.info("Starting preprocessing pipeline...")
    
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    
    # Verify no missing values remain (except for columns we'll handle later)
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Remaining missing values: {remaining_missing}")
    else:
        logger.info("All missing values handled successfully")
    
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_types(df: pd.DataFrame) -> dict:
    """
    Categorize columns by data type.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with 'numerical' and 'categorical' column lists
    """
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    return {
        "numerical": numerical,
        "categorical": categorical
    }


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform data quality checks.
    
    Args:
        df: DataFrame to check
    
    Returns:
        Dictionary with quality metrics
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_missing": df.isnull().sum().sum(),
        "missing_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
