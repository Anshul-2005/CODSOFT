"""
Prediction module for Movie Rating Prediction.
Provides inference capabilities for trained models.

Author: CodSoft Data Science Intern
Task: Task 2 - Movie Rating Prediction
"""

from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from src.utils import MODELS_DIR, OUTPUTS_DIR, verify_directory, print_section

logger = logging.getLogger("MovieRatingPrediction")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(filename: str = "movie_rating_model.pkl") -> Any:
    """
    Load the saved trained model.
    
    Args:
        filename: Model filename
    
    Returns:
        Loaded model
    """
    filepath = MODELS_DIR / filename
    
    if not filepath.exists():
        error_msg = (
            f"Model not found at: {filepath}\n"
            f"Please run 'python main.py' first to train and save the model."
        )
        logger.error(f"Model not found: {filepath}")
        raise FileNotFoundError(error_msg)
    
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ============================================================================
# SINGLE PREDICTION
# ============================================================================

def predict_rating(
    model: Any,
    feature_names: List[str],
    movie_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Predict rating for a single movie.
    
    Args:
        model: Trained model
        feature_names: List of expected feature names
        movie_data: Dictionary of movie features
        **kwargs: Alternative way to pass features
    
    Returns:
        Dictionary with predicted rating
    """
    data = movie_data or {}
    data.update(kwargs)
    
    if not data:
        raise ValueError("No movie data provided")
    
    # Build feature vector
    input_data = {}
    for feat in feature_names:
        if feat in data:
            input_data[feat] = data[feat]
        else:
            input_data[feat] = 0
            logger.debug(f"Missing feature '{feat}', using default value 0")
    
    X = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Clip to valid rating range
    prediction = np.clip(prediction, 0, 10)
    
    result = {
        "Predicted_Rating": round(float(prediction), 2),
        "Rating_Category": _get_rating_category(prediction),
    }
    
    return result


def _get_rating_category(rating: float) -> str:
    """Categorize rating into descriptive label."""
    if rating >= 8.0:
        return "Excellent"
    elif rating >= 7.0:
        return "Good"
    elif rating >= 6.0:
        return "Average"
    elif rating >= 5.0:
        return "Below Average"
    else:
        return "Poor"


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def generate_predictions_csv(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    filename: str = "predictions.csv",
) -> Path:
    """
    Generate predictions CSV for the test set.
    
    Output columns:
        - Actual: True ratings
        - Predicted: Model predictions
        - Residual: Actual - Predicted
        - AbsoluteError: |Actual - Predicted|
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        filename: Output filename
    
    Returns:
        Path to saved CSV file
    """
    verify_directory(OUTPUTS_DIR)
    
    y_pred = model.predict(X_test)
    
    predictions_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": np.round(y_pred, 2),
        "Residual": np.round(y_test.values - y_pred, 2),
        "AbsoluteError": np.round(np.abs(y_test.values - y_pred), 2),
    })
    
    filepath = OUTPUTS_DIR / filename
    predictions_df.to_csv(filepath, index=False)
    
    # Log summary
    mae = predictions_df["AbsoluteError"].mean()
    within_05 = (predictions_df["AbsoluteError"] <= 0.5).mean() * 100
    within_1 = (predictions_df["AbsoluteError"] <= 1.0).mean() * 100
    
    logger.info(f"Predictions saved: {filepath}")
    logger.info(f"  MAE: {mae:.3f}")
    logger.info(f"  Within ±0.5: {within_05:.1f}%")
    logger.info(f"  Within ±1.0: {within_1:.1f}%")
    
    return filepath


# ============================================================================
# DEMO PREDICTIONS
# ============================================================================

def demo_predictions(model: Any, feature_names: List[str]) -> None:
    """
    Run sample predictions to demonstrate the model.
    
    Demonstrates predictions for example movies with different characteristics.
    
    Args:
        model: Trained model
        feature_names: List of expected feature names
    """
    print_section("SAMPLE PREDICTIONS")
    
    # Example 1: Popular action movie
    movie_1 = {
        "Year": 2020,
        "Duration_Minutes": 150,
        "Votes": 100000,
        "Votes_Log": np.log1p(100000),
        "PrimaryGenre_Encoded": 0,  # Action
        "GenreCount": 3,
        "Director_Encoded": 1,
        "Director_AvgRating": 7.5,
        "LeadActor_Encoded": 1,
        "LeadActor_AvgRating": 7.2,
        "Decade": 2020,
        "MovieAge": 4,
        "VoteCategory_Encoded": 4,  # VeryHigh
        "DurationCategory_Encoded": 3,  # VeryLong
    }
    
    result1 = predict_rating(model, feature_names, movie_1)
    print(f"\n  Movie 1: Popular Action Movie (2020)")
    print(f"    - Duration: 150 min, Votes: 100,000")
    print(f"    Predicted Rating: {result1['Predicted_Rating']}")
    print(f"    Category: {result1['Rating_Category']}")
    
    # Example 2: Indie drama
    movie_2 = {
        "Year": 2015,
        "Duration_Minutes": 100,
        "Votes": 5000,
        "Votes_Log": np.log1p(5000),
        "PrimaryGenre_Encoded": 5,  # Drama
        "GenreCount": 2,
        "Director_Encoded": 10,
        "Director_AvgRating": 6.8,
        "LeadActor_Encoded": 20,
        "LeadActor_AvgRating": 6.5,
        "Decade": 2010,
        "MovieAge": 9,
        "VoteCategory_Encoded": 2,  # Medium
        "DurationCategory_Encoded": 1,  # Standard
    }
    
    result2 = predict_rating(model, feature_names, movie_2)
    print(f"\n  Movie 2: Indie Drama (2015)")
    print(f"    - Duration: 100 min, Votes: 5,000")
    print(f"    Predicted Rating: {result2['Predicted_Rating']}")
    print(f"    Category: {result2['Rating_Category']}")
    
    # Example 3: Old classic
    movie_3 = {
        "Year": 1980,
        "Duration_Minutes": 180,
        "Votes": 500000,
        "Votes_Log": np.log1p(500000),
        "PrimaryGenre_Encoded": 5,  # Drama
        "GenreCount": 2,
        "Director_Encoded": 5,
        "Director_AvgRating": 8.2,
        "LeadActor_Encoded": 5,
        "LeadActor_AvgRating": 7.8,
        "Decade": 1980,
        "MovieAge": 44,
        "VoteCategory_Encoded": 4,  # VeryHigh
        "DurationCategory_Encoded": 3,  # VeryLong
    }
    
    result3 = predict_rating(model, feature_names, movie_3)
    print(f"\n  Movie 3: Classic Epic (1980)")
    print(f"    - Duration: 180 min, Votes: 500,000")
    print(f"    Predicted Rating: {result3['Predicted_Rating']}")
    print(f"    Category: {result3['Rating_Category']}")


# ============================================================================
# PREDICTION UTILITIES
# ============================================================================

def get_prediction_summary(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Get summary statistics of predictions.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with summary statistics
    """
    residuals = y_test.values - y_pred
    abs_errors = np.abs(residuals)
    
    return {
        "mean_actual": y_test.mean(),
        "mean_predicted": y_pred.mean(),
        "mean_residual": residuals.mean(),
        "std_residual": residuals.std(),
        "mae": abs_errors.mean(),
        "max_error": abs_errors.max(),
        "within_0.5": (abs_errors <= 0.5).mean(),
        "within_1.0": (abs_errors <= 1.0).mean(),
    }
