"""
Feature engineering module for Titanic Survival Prediction.
Creates new predictive features from existing data.

Author: CodSoft Data Science Intern
Task: Task 1 - Titanic Survival Prediction
"""

from typing import List, Optional
import logging

import pandas as pd
import numpy as np

# Get logger from utils
logger = logging.getLogger("TitanicPrediction")


# ============================================================================
# TITLE EXTRACTION
# ============================================================================

def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract passenger title from Name column.
    
    Why This Feature Matters:
        - Titles encode social status, gender, and approximate age
        - Strong predictors: 'Mrs' (married woman), 'Master' (young boy)
        - Examples: "Braund, Mr. Owen Harris" → "Mr"
    
    Consolidation:
        - Rare titles (Dr, Rev, Col, etc.) grouped as 'Rare'
        - French titles mapped to English equivalents
    
    Args:
        df: DataFrame with 'Name' column
    
    Returns:
        DataFrame with new 'Title' column
    """
    df = df.copy()
    
    if "Name" not in df.columns:
        logger.warning("Name column not found - skipping title extraction")
        return df
    
    # Extract title using regex: finds word before period
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    
    # Consolidate rare titles into common groups
    title_mapping = {
        # Common titles
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        # French titles
        "Mlle": "Miss",      # Mademoiselle
        "Mme": "Mrs",        # Madame
        "Ms": "Miss",
        # Rare/Professional titles
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Capt": "Rare",
        # Nobility titles
        "Countess": "Rare",
        "Lady": "Rare",
        "Sir": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Jonkheer": "Rare",
    }
    
    df["Title"] = df["Title"].map(title_mapping).fillna("Rare")
    
    logger.info(f"Feature created: Title (categories: {df['Title'].nunique()})")
    return df


# ============================================================================
# FAMILY FEATURES
# ============================================================================

def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create FamilySize = SibSp + Parch + 1.
    
    Why This Feature Matters:
        - Small families (2-4 members) had better survival rates
        - Solo travelers and very large families had lower survival
        - Captures family dynamics during evacuation
    
    Args:
        df: DataFrame with 'SibSp' and 'Parch' columns
    
    Returns:
        DataFrame with new 'FamilySize' column
    """
    df = df.copy()
    
    if "SibSp" not in df.columns or "Parch" not in df.columns:
        logger.warning("SibSp/Parch columns not found - skipping FamilySize")
        return df
    
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    logger.info(f"Feature created: FamilySize (range: {df['FamilySize'].min()}-{df['FamilySize'].max()})")
    return df


def create_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create IsAlone flag (1 if traveling alone, 0 otherwise).
    
    Why This Feature Matters:
        - Solo travelers had lower survival rates
        - Binary flag simplifies the family size signal
        - Easier for linear models to interpret
    
    Args:
        df: DataFrame with 'FamilySize' column
    
    Returns:
        DataFrame with new 'IsAlone' column
    """
    df = df.copy()
    
    if "FamilySize" not in df.columns:
        # Create FamilySize if not exists
        df = create_family_size(df)
    
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    
    alone_count = df["IsAlone"].sum()
    logger.info(f"Feature created: IsAlone ({alone_count}/{len(df)} traveling alone)")
    return df


# ============================================================================
# FARE FEATURES
# ============================================================================

def create_fare_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Fare into categories: Low, Medium, High, VeryHigh.
    
    Why This Feature Matters:
        - Fare is a proxy for socio-economic status
        - Higher fares = upper-class cabins = closer to lifeboats
        - Binning reduces noise from extreme outliers
    
    Uses quantile-based binning (quartiles) for balanced groups.
    
    Args:
        df: DataFrame with 'Fare' column
    
    Returns:
        DataFrame with new 'FareCategory' column
    """
    df = df.copy()
    
    if "Fare" not in df.columns:
        logger.warning("Fare column not found - skipping FareCategory")
        return df
    
    try:
        df["FareCategory"] = pd.qcut(
            df["Fare"], 
            q=4, 
            labels=["Low", "Medium", "High", "VeryHigh"],
            duplicates="drop"
        ).astype(str)
    except ValueError:
        # Fallback to cut if qcut fails due to duplicates
        df["FareCategory"] = pd.cut(
            df["Fare"],
            bins=4,
            labels=["Low", "Medium", "High", "VeryHigh"]
        ).astype(str)
    
    logger.info("Feature created: FareCategory (4 bins)")
    return df


# ============================================================================
# AGE FEATURES
# ============================================================================

def create_age_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Age into categories: Child, Teen, Adult, Senior.
    
    Why This Feature Matters:
        - Children had priority for lifeboats ('women and children first')
        - Age groups capture non-linear survival patterns
        - Handles the non-linear relationship between age and survival
    
    Bins:
        - Child: 0-12
        - Teen: 13-18
        - Adult: 19-60
        - Senior: 60+
    
    Args:
        df: DataFrame with 'Age' column
    
    Returns:
        DataFrame with new 'AgeCategory' column
    """
    df = df.copy()
    
    if "Age" not in df.columns:
        logger.warning("Age column not found - skipping AgeCategory")
        return df
    
    bins = [0, 12, 18, 60, 120]
    labels = ["Child", "Teen", "Adult", "Senior"]
    
    df["AgeCategory"] = pd.cut(
        df["Age"], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    ).astype(str)
    
    logger.info(f"Feature created: AgeCategory (4 bins)")
    return df


# ============================================================================
# CABIN FEATURES
# ============================================================================

def create_cabin_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CabinAvailable flag (1 if cabin number recorded, 0 otherwise).
    
    Why This Feature Matters:
        - Having a cabin number recorded may indicate higher-class passengers
        - Better documentation = closer to crew = better survival chances
        - Proxy for socio-economic status and cabin location
    
    Args:
        df: DataFrame with 'Cabin' column
    
    Returns:
        DataFrame with new 'CabinAvailable' column
    """
    df = df.copy()
    
    if "Cabin" not in df.columns:
        logger.warning("Cabin column not found - skipping CabinAvailable")
        return df
    
    df["CabinAvailable"] = (
        (df["Cabin"] != "Unknown") & 
        (df["Cabin"].notna()) & 
        (df["Cabin"] != "")
    ).astype(int)
    
    available_count = df["CabinAvailable"].sum()
    logger.info(f"Feature created: CabinAvailable ({available_count}/{len(df)} with cabin)")
    return df


# ============================================================================
# TICKET FEATURES
# ============================================================================

def create_ticket_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many passengers share the same ticket number.
    
    Why This Feature Matters:
        - Shared tickets indicate families or groups traveling together
        - Group size affects survival probability
        - Captures group coordination and shared resources
    
    Args:
        df: DataFrame with 'Ticket' column
    
    Returns:
        DataFrame with new 'TicketFrequency' column
    """
    df = df.copy()
    
    if "Ticket" not in df.columns:
        logger.warning("Ticket column not found - skipping TicketFrequency")
        return df
    
    ticket_counts = df["Ticket"].value_counts()
    df["TicketFrequency"] = df["Ticket"].map(ticket_counts)
    
    max_freq = df["TicketFrequency"].max()
    logger.info(f"Feature created: TicketFrequency (max shared: {max_freq})")
    return df


# ============================================================================
# ENCODING ENGINEERED FEATURES
# ============================================================================

def encode_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the engineered categorical features using one-hot encoding.
    
    Encodes:
        - FareCategory
        - AgeCategory
    
    Args:
        df: DataFrame with engineered categorical features
    
    Returns:
        DataFrame with one-hot encoded features
    """
    df = df.copy()
    
    categorical_cols = ["FareCategory", "AgeCategory"]
    
    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
            logger.info(f"{col}: One-hot encoded into {len(dummies.columns)} columns")
    
    return df


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    
    Pipeline Steps:
        1. Extract Title from Name
        2. Create FamilySize
        3. Create IsAlone
        4. Create CabinAvailable
        5. Create TicketFrequency
        6. Create FareCategory
        7. Create AgeCategory
        8. Encode categorical engineered features
    
    Note: Must be called BEFORE encoding and column dropping.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Create features
    df = extract_title(df)
    df = create_family_size(df)
    df = create_is_alone(df)
    df = create_cabin_available(df)
    df = create_ticket_frequency(df)
    df = create_fare_category(df)
    df = create_age_category(df)
    
    # Encode categorical features
    df = encode_engineered_features(df)
    
    logger.info(f"Feature engineering complete. New shape: {df.shape}")
    return df


# ============================================================================
# FEATURE LIST
# ============================================================================

def get_engineered_feature_names() -> List[str]:
    """
    Return list of all engineered feature names.
    
    Returns:
        List of feature names created by this module
    """
    return [
        "Title",
        "FamilySize",
        "IsAlone",
        "CabinAvailable",
        "TicketFrequency",
        "FareCategory_Low",
        "FareCategory_Medium",
        "FareCategory_High",
        "FareCategory_VeryHigh",
        "AgeCategory_Child",
        "AgeCategory_Teen",
        "AgeCategory_Adult",
        "AgeCategory_Senior",
    ]
