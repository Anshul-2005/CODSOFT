"""
Data preprocessing module for Movie Rating Prediction.
Handles missing values, data cleaning, encoding, and scaling.

Author: CodSoft Data Science Intern
Task: Task 2 - Movie Rating Prediction
"""

from typing import Tuple, Optional, List
import logging
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger("MovieRatingPrediction")


# ============================================================================
# DATA TYPE CLEANING
# ============================================================================

def clean_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and extract year from the Year column.
    
    Handles:
        - Year ranges like "2019-2020" (takes first year)
        - Parentheses like "(2020)"
        - Non-numeric characters
    
    Args:
        df: DataFrame with 'Year' column
    
    Returns:
        DataFrame with cleaned 'Year' column as numeric
    """
    df = df.copy()
    
    if "Year" not in df.columns:
        logger.warning("Year column not found")
        return df
    
    def extract_year(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        # Remove parentheses
        val_str = val_str.replace("(", "").replace(")", "")
        # Extract first 4-digit year
        match = re.search(r'(\d{4})', val_str)
        if match:
            return int(match.group(1))
        return np.nan
    
    df["Year"] = df["Year"].apply(extract_year)
    
    # Filter reasonable years (1900-2030)
    df.loc[(df["Year"] < 1900) | (df["Year"] > 2030), "Year"] = np.nan
    
    valid_years = df["Year"].notna().sum()
    logger.info(f"Year: Cleaned {valid_years}/{len(df)} valid entries")
    
    return df


def clean_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and extract duration in minutes.
    
    Handles:
        - "120 min" format
        - Numeric values
        - Missing/invalid values
    
    Args:
        df: DataFrame with 'Duration' column
    
    Returns:
        DataFrame with 'Duration_Minutes' as numeric
    """
    df = df.copy()
    
    if "Duration" not in df.columns:
        logger.warning("Duration column not found")
        return df
    
    def extract_minutes(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().lower()
        # Remove "min" and extract number
        match = re.search(r'(\d+)', val_str)
        if match:
            minutes = int(match.group(1))
            # Filter reasonable durations (1-600 minutes)
            if 1 <= minutes <= 600:
                return minutes
        return np.nan
    
    df["Duration_Minutes"] = df["Duration"].apply(extract_minutes)
    
    valid_durations = df["Duration_Minutes"].notna().sum()
    logger.info(f"Duration: Extracted {valid_durations}/{len(df)} valid entries")
    
    return df


def clean_votes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert votes to numeric.
    
    Handles:
        - Comma separators (1,234,567)
        - Invalid/missing values
    
    Args:
        df: DataFrame with 'Votes' column
    
    Returns:
        DataFrame with 'Votes' as numeric
    """
    df = df.copy()
    
    if "Votes" not in df.columns:
        logger.warning("Votes column not found")
        return df
    
    def extract_votes(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        # Remove commas and non-numeric chars
        val_str = re.sub(r'[^\d]', '', val_str)
        if val_str:
            return int(val_str)
        return np.nan
    
    df["Votes"] = df["Votes"].apply(extract_votes)
    
    valid_votes = df["Votes"].notna().sum()
    logger.info(f"Votes: Cleaned {valid_votes}/{len(df)} valid entries")
    
    return df


def clean_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Rating column (target variable).
    
    Args:
        df: DataFrame with 'Rating' column
    
    Returns:
        DataFrame with cleaned 'Rating' column
    """
    df = df.copy()
    
    if "Rating" not in df.columns:
        logger.warning("Rating column not found")
        return df
    
    # Convert to numeric, coercing errors to NaN
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    
    # Filter valid ratings (0-10 scale)
    df.loc[(df["Rating"] < 0) | (df["Rating"] > 10), "Rating"] = np.nan
    
    valid_ratings = df["Rating"].notna().sum()
    logger.info(f"Rating: {valid_ratings}/{len(df)} valid entries")
    
    return df


# ============================================================================
# MISSING VALUE HANDLING
# ============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Strategy:
        - Rating: Drop rows (target variable)
        - Duration_Minutes: Fill with median
        - Votes: Fill with median
        - Year: Fill with median
        - Categorical: Fill with 'Unknown'
    
    Args:
        df: DataFrame with missing values
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    # Drop rows with missing target (Rating)
    if "Rating" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Rating"])
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Rating: Dropped {dropped} rows with missing target")
    
    # Numeric columns: median imputation
    numeric_cols = ["Duration_Minutes", "Votes", "Year"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"{col}: Imputed {missing_count} values with median ({median_val:.1f})")
    
    # Categorical columns: fill with 'Unknown'
    categorical_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3", "Name"]
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            df[col].fillna("Unknown", inplace=True)
            logger.info(f"{col}: Filled {missing_count} values with 'Unknown'")
    
    return df


# ============================================================================
# DUPLICATE HANDLING
# ============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
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
# OUTLIER HANDLING
# ============================================================================

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = "iqr") -> pd.DataFrame:
    """
    Handle outliers in specified columns.
    
    Args:
        df: DataFrame
        columns: List of column names to check
        method: 'iqr' (clip to IQR bounds) or 'zscore' (remove > 3 std)
    
    Returns:
        DataFrame with handled outliers
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            
            if outliers > 0:
                logger.info(f"{col}: Clipped {outliers} outliers to IQR bounds")
        
        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            outliers = (z_scores > 3).sum()
            df = df[z_scores <= 3]
            
            if outliers > 0:
                logger.info(f"{col}: Removed {outliers} outliers (z-score > 3)")
    
    return df


# ============================================================================
# WHITESPACE CLEANING
# ============================================================================

def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean whitespace from string columns.
    """
    df = df.copy()
    
    string_cols = df.select_dtypes(include=["object"]).columns
    
    for col in string_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    logger.info(f"Cleaned whitespace in {len(string_cols)} string columns")
    return df


# ============================================================================
# FEATURE SCALING
# ============================================================================

def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to normalize numerical features.
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
    """Apply a pre-fitted scaler to new data."""
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
        1. Clean whitespace
        2. Clean Year column
        3. Clean Duration column
        4. Clean Votes column
        5. Clean Rating column (target)
        6. Handle missing values
        7. Remove duplicates
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting preprocessing pipeline...")
    
    df = clean_whitespace(df)
    df = clean_year(df)
    df = clean_duration(df)
    df = clean_votes(df)
    df = clean_rating(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    
    remaining_missing = df.isnull().sum().sum()
    logger.info(f"Preprocessing complete. Remaining missing values: {remaining_missing}")
    logger.info(f"Final shape: {df.shape}")
    
    return df
