"""
Feature Engineering Module
===========================

Handles feature creation and transformation for the
Credit Card Fraud Detection project.

Transformations:
    1. Time-based features (Hour of Day, Time Period)
    2. Amount-based features (Log Amount, Amount Bins)
    3. V-feature interactions (statistical aggregations)
    4. Anomaly score features

All transformations are documented and justified.
"""

import logging
from typing import Tuple, List

import pandas as pd
import numpy as np

# Logger
logger = logging.getLogger("CreditCardFraudDetection")


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the 'Time' column.
    
    The 'Time' column represents seconds elapsed since the first
    transaction. We extract cyclical hour features.
    
    Transformations:
        - Hour: Hour of the day (Time mod 86400 / 3600)
        - Is_Night: Whether the transaction occurred at night (22:00-06:00)
    
    Args:
        df: Input DataFrame with 'Time' column
        
    Returns:
        pd.DataFrame: DataFrame with new time features
    """
    df = df.copy()
    
    # Extract hour of day (cyclical - transactions span ~48 hours)
    df["Hour"] = (df["Time"] % 86400) / 3600
    df["Hour"] = df["Hour"].astype(int)
    
    # Night transactions (10 PM to 6 AM) — fraud may spike at unusual hours
    df["Is_Night"] = ((df["Hour"] >= 22) | (df["Hour"] < 6)).astype(int)
    
    logger.info("Time features created: Hour, Is_Night")
    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create amount-based features from the 'Amount' column.
    
    Transformations:
        - Log_Amount: Log-transformed amount (reduces skewness)
        - Amount_Bin: Binned amount categories
    
    Args:
        df: Input DataFrame with 'Amount' column
        
    Returns:
        pd.DataFrame: DataFrame with new amount features
    """
    df = df.copy()
    
    # Log transformation to reduce skewness of Amount
    df["Log_Amount"] = np.log1p(df["Amount"])
    
    # Bin amounts into categories
    bins = [-1, 5, 50, 200, 1000, np.inf]
    labels = [0, 1, 2, 3, 4]  # Very Low, Low, Medium, High, Very High
    df["Amount_Bin"] = pd.cut(df["Amount"], bins=bins, labels=labels).astype(int)
    
    logger.info("Amount features created: Log_Amount, Amount_Bin")
    return df


def create_v_feature_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistical aggregations from V1-V28 PCA features.
    
    These features capture the overall pattern across PCA components,
    which can help detect anomalous transactions.
    
    Transformations:
        - V_Mean: Mean of V1-V28
        - V_Std: Standard deviation of V1-V28
        - V_Skew: Number of V features with absolute value > 2
    
    Args:
        df: Input DataFrame with V1-V28 columns
        
    Returns:
        pd.DataFrame: DataFrame with aggregated V features
    """
    df = df.copy()
    
    v_columns = [f"V{i}" for i in range(1, 29)]
    
    # Mean across all V features
    df["V_Mean"] = df[v_columns].mean(axis=1)
    
    # Standard deviation across V features (high std = unusual pattern)
    df["V_Std"] = df[v_columns].std(axis=1)
    
    # Count of extreme V values (potential anomaly indicator)
    df["V_Extreme_Count"] = (df[v_columns].abs() > 2).sum(axis=1)
    
    logger.info("V-feature aggregations created: V_Mean, V_Std, V_Extreme_Count")
    return df


def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Apply all feature engineering transformations.
    
    This function applies the same transformations to both training
    and testing sets to ensure consistency.
    
    Args:
        X_train: Training features
        X_test: Testing features
        
    Returns:
        Tuple of (transformed X_train, transformed X_test, list of new features)
    """
    logger.info("Starting feature engineering pipeline...")
    
    original_cols = list(X_train.columns)
    
    # Step 1: Time features
    X_train = create_time_features(X_train)
    X_test = create_time_features(X_test)
    
    # Step 2: Amount features
    X_train = create_amount_features(X_train)
    X_test = create_amount_features(X_test)
    
    # Step 3: V-feature aggregations
    X_train = create_v_feature_aggregations(X_train)
    X_test = create_v_feature_aggregations(X_test)
    
    # Identify new features
    new_features = [col for col in X_train.columns if col not in original_cols]
    
    logger.info(f"Feature engineering completed | New features: {len(new_features)}")
    logger.info(f"New features: {new_features}")
    logger.info(f"Total features: {X_train.shape[1]}")
    
    return X_train, X_test, new_features
