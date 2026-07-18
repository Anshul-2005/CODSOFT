"""
Feature engineering module for Movie Rating Prediction.
Creates and transforms features for model training.

Author: CodSoft Data Science Intern
Task: Task 2 - Movie Rating Prediction
"""

from typing import List, Tuple, Dict, Any
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("MovieRatingPrediction")


# ============================================================================
# GENRE ENCODING
# ============================================================================

def encode_primary_genre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and encode the primary genre.
    
    Movies often have multiple genres (e.g., "Action, Drama, Thriller").
    We extract the first genre as the primary genre.
    
    Args:
        df: DataFrame with 'Genre' column
    
    Returns:
        DataFrame with 'PrimaryGenre' and 'PrimaryGenre_Encoded' columns
    """
    df = df.copy()
    
    if "Genre" not in df.columns:
        logger.warning("Genre column not found")
        return df
    
    # Extract primary genre (first genre listed)
    df["PrimaryGenre"] = df["Genre"].apply(
        lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "Unknown"
    )
    
    # Label encode
    le = LabelEncoder()
    df["PrimaryGenre_Encoded"] = le.fit_transform(df["PrimaryGenre"])
    
    n_genres = df["PrimaryGenre"].nunique()
    logger.info(f"PrimaryGenre: Extracted {n_genres} unique genres")
    
    return df


def count_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of genres per movie.
    
    Why: Movies with multiple genres may have different rating patterns.
    
    Args:
        df: DataFrame with 'Genre' column
    
    Returns:
        DataFrame with 'GenreCount' column
    """
    df = df.copy()
    
    if "Genre" not in df.columns:
        logger.warning("Genre column not found")
        return df
    
    def count_genre_parts(x):
        if pd.isna(x) or str(x).strip() == "":
            return 0
        return len(str(x).split(","))
    
    df["GenreCount"] = df["Genre"].apply(count_genre_parts)
    
    logger.info(f"GenreCount: Range {df['GenreCount'].min()}-{df['GenreCount'].max()}")
    return df


# ============================================================================
# DIRECTOR ENCODING
# ============================================================================

def encode_director_popularity(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Encode directors based on their movie count.
    
    Strategy:
        - Top N directors get unique encoding
        - Others encoded as 'Other'
    
    Args:
        df: DataFrame with 'Director' column
        top_n: Number of top directors to keep
    
    Returns:
        DataFrame with 'Director_Encoded' column
    """
    df = df.copy()
    
    if "Director" not in df.columns:
        logger.warning("Director column not found")
        return df
    
    # Get director counts
    director_counts = df["Director"].value_counts()
    top_directors = director_counts.head(top_n).index.tolist()
    
    # Create director category
    df["Director_Category"] = df["Director"].apply(
        lambda x: x if x in top_directors else "Other"
    )
    
    # Label encode
    le = LabelEncoder()
    df["Director_Encoded"] = le.fit_transform(df["Director_Category"])
    
    logger.info(f"Director: Top {top_n} directors encoded, others as 'Other'")
    return df


def calculate_director_avg_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average rating per director.
    
    Why: Some directors consistently produce higher-rated movies.
    
    Args:
        df: DataFrame with 'Director' and 'Rating' columns
    
    Returns:
        DataFrame with 'Director_AvgRating' column
    """
    df = df.copy()
    
    if "Director" not in df.columns or "Rating" not in df.columns:
        logger.warning("Director or Rating column not found")
        return df
    
    # Calculate mean rating per director
    director_avg = df.groupby("Director")["Rating"].mean().to_dict()
    df["Director_AvgRating"] = df["Director"].map(director_avg)
    
    # Fill missing with global mean
    global_mean = df["Rating"].mean()
    df["Director_AvgRating"].fillna(global_mean, inplace=True)
    
    logger.info(f"Director_AvgRating: Calculated (global mean: {global_mean:.2f})")
    return df


# ============================================================================
# ACTOR ENCODING
# ============================================================================

def encode_lead_actor(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Encode lead actor (Actor 1) based on popularity.
    
    Args:
        df: DataFrame with 'Actor 1' column
        top_n: Number of top actors to keep
    
    Returns:
        DataFrame with 'LeadActor_Encoded' column
    """
    df = df.copy()
    
    actor_col = "Actor 1"
    if actor_col not in df.columns:
        logger.warning(f"{actor_col} column not found")
        return df
    
    # Get actor counts
    actor_counts = df[actor_col].value_counts()
    top_actors = actor_counts.head(top_n).index.tolist()
    
    # Create actor category
    df["LeadActor_Category"] = df[actor_col].apply(
        lambda x: x if x in top_actors else "Other"
    )
    
    # Label encode
    le = LabelEncoder()
    df["LeadActor_Encoded"] = le.fit_transform(df["LeadActor_Category"])
    
    logger.info(f"LeadActor: Top {top_n} actors encoded")
    return df


def calculate_actor_avg_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average rating for lead actor.
    
    Args:
        df: DataFrame with 'Actor 1' and 'Rating' columns
    
    Returns:
        DataFrame with 'LeadActor_AvgRating' column
    """
    df = df.copy()
    
    actor_col = "Actor 1"
    if actor_col not in df.columns or "Rating" not in df.columns:
        logger.warning(f"{actor_col} or Rating column not found")
        return df
    
    # Calculate mean rating per actor
    actor_avg = df.groupby(actor_col)["Rating"].mean().to_dict()
    df["LeadActor_AvgRating"] = df[actor_col].map(actor_avg)
    
    # Fill missing with global mean
    global_mean = df["Rating"].mean()
    df["LeadActor_AvgRating"].fillna(global_mean, inplace=True)
    
    logger.info(f"LeadActor_AvgRating: Calculated")
    return df


# ============================================================================
# TEMPORAL FEATURES
# ============================================================================

def create_decade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create decade feature from year.
    
    Why: Movie rating patterns may vary by decade.
    
    Args:
        df: DataFrame with 'Year' column
    
    Returns:
        DataFrame with 'Decade' column
    """
    df = df.copy()
    
    if "Year" not in df.columns:
        logger.warning("Year column not found")
        return df
    
    df["Decade"] = (df["Year"] // 10 * 10).astype(int)
    
    logger.info(f"Decade: Created (range: {df['Decade'].min()}-{df['Decade'].max()})")
    return df


def create_movie_age(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    """
    Calculate movie age (years since release).
    
    Args:
        df: DataFrame with 'Year' column
        current_year: Reference year
    
    Returns:
        DataFrame with 'MovieAge' column
    """
    df = df.copy()
    
    if "Year" not in df.columns:
        logger.warning("Year column not found")
        return df
    
    df["MovieAge"] = current_year - df["Year"]
    df["MovieAge"] = df["MovieAge"].clip(lower=0)  # No negative ages
    
    logger.info(f"MovieAge: Created (range: {df['MovieAge'].min()}-{df['MovieAge'].max()})")
    return df


# ============================================================================
# VOTE FEATURES
# ============================================================================

def create_vote_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin votes into popularity categories.
    
    Categories:
        - Very Low: < 100
        - Low: 100-1000
        - Medium: 1000-10000
        - High: 10000-100000
        - Very High: > 100000
    
    Args:
        df: DataFrame with 'Votes' column
    
    Returns:
        DataFrame with 'VoteCategory' and 'VoteCategory_Encoded' columns
    """
    df = df.copy()
    
    if "Votes" not in df.columns:
        logger.warning("Votes column not found")
        return df
    
    bins = [0, 100, 1000, 10000, 100000, float('inf')]
    labels = ["VeryLow", "Low", "Medium", "High", "VeryHigh"]
    
    df["VoteCategory"] = pd.cut(df["Votes"], bins=bins, labels=labels)
    
    # Label encode
    le = LabelEncoder()
    df["VoteCategory_Encoded"] = le.fit_transform(df["VoteCategory"].astype(str))
    
    logger.info(f"VoteCategory: Created 5 categories")
    return df


def log_transform_votes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to votes.
    
    Why: Votes are highly skewed; log transform normalizes distribution.
    
    Args:
        df: DataFrame with 'Votes' column
    
    Returns:
        DataFrame with 'Votes_Log' column
    """
    df = df.copy()
    
    if "Votes" not in df.columns:
        logger.warning("Votes column not found")
        return df
    
    # Log1p handles zeros
    df["Votes_Log"] = np.log1p(df["Votes"])
    
    logger.info(f"Votes_Log: Log-transformed")
    return df


# ============================================================================
# DURATION FEATURES
# ============================================================================

def create_duration_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin duration into categories.
    
    Categories:
        - Short: < 90 min
        - Standard: 90-120 min
        - Long: 120-150 min
        - VeryLong: > 150 min
    
    Args:
        df: DataFrame with 'Duration_Minutes' column
    
    Returns:
        DataFrame with 'DurationCategory' column
    """
    df = df.copy()
    
    if "Duration_Minutes" not in df.columns:
        logger.warning("Duration_Minutes column not found")
        return df
    
    bins = [0, 90, 120, 150, float('inf')]
    labels = ["Short", "Standard", "Long", "VeryLong"]
    
    df["DurationCategory"] = pd.cut(df["Duration_Minutes"], bins=bins, labels=labels)
    
    # Label encode
    le = LabelEncoder()
    df["DurationCategory_Encoded"] = le.fit_transform(df["DurationCategory"].astype(str))
    
    logger.info(f"DurationCategory: Created 4 categories")
    return df


# ============================================================================
# DROP UNUSED COLUMNS
# ============================================================================

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns not needed for modeling.
    
    Dropped:
        - Name: Just identifier
        - Duration: Replaced by Duration_Minutes
        - Genre: Replaced by engineered features
        - Director: Replaced by encoded features
        - Actor 1/2/3: Replaced by encoded features
        - Category columns: Keep only encoded versions
    
    Args:
        df: DataFrame with all columns
    
    Returns:
        DataFrame with necessary columns only
    """
    df = df.copy()
    
    cols_to_drop = [
        "Name", "Duration", "Genre", "Director",
        "Actor 1", "Actor 2", "Actor 3",
        "PrimaryGenre", "Director_Category", "LeadActor_Category",
        "VoteCategory", "DurationCategory"
    ]
    
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    
    if existing:
        logger.info(f"Dropped columns: {existing}")
    
    return df


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def full_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    
    Pipeline:
        1. Genre encoding
        2. Genre count
        3. Director encoding
        4. Director avg rating
        5. Actor encoding
        6. Actor avg rating
        7. Decade
        8. Movie age
        9. Vote category
        10. Log votes
        11. Duration category
        12. Drop unused columns
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    df = encode_primary_genre(df)
    df = count_genres(df)
    df = encode_director_popularity(df)
    df = calculate_director_avg_rating(df)
    df = encode_lead_actor(df)
    df = calculate_actor_avg_rating(df)
    df = create_decade(df)
    df = create_movie_age(df)
    df = create_vote_category(df)
    df = log_transform_votes(df)
    df = create_duration_category(df)
    df = drop_unused_columns(df)
    
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame, target: str = "Rating") -> List[str]:
    """
    Get list of feature columns (excluding target).
    
    Args:
        df: DataFrame
        target: Target column name
    
    Returns:
        List of feature column names
    """
    return [col for col in df.columns if col != target]
