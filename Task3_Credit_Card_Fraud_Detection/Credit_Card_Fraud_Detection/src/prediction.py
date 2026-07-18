"""
Prediction Module
==================

Handles making predictions and generating prediction outputs
for the Credit Card Fraud Detection project.

Functions:
    - predict: Make predictions using trained model
    - generate_predictions_csv: Generate predictions.csv output
    - load_saved_model: Load a saved model from disk
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
import joblib

from src.utils import get_project_root, save_dataframe

# Logger
logger = logging.getLogger("CreditCardFraudDetection")


def predict(
    model: Any,
    X: pd.DataFrame
) -> tuple:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model instance
        X: Feature DataFrame
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    predictions = model.predict(X)
    
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X)
    
    logger.info(f"Predictions generated | Total: {len(predictions)}")
    logger.info(f"Predicted Fraud: {sum(predictions)} | Predicted Legitimate: {len(predictions) - sum(predictions)}")
    
    return predictions, probabilities


def generate_predictions_csv(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    save: bool = True
) -> pd.DataFrame:
    """
    Generate predictions CSV with actual, predicted, and probability columns.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        save: Whether to save to file
        
    Returns:
        pd.DataFrame: Predictions DataFrame
    """
    predictions_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
    })
    
    if y_prob is not None:
        predictions_df["Probability"] = np.round(y_prob, 6)
    
    # Add correct/incorrect flag
    predictions_df["Correct"] = (
        predictions_df["Actual"] == predictions_df["Predicted"]
    ).astype(int)
    
    # Summary
    total = len(predictions_df)
    correct = predictions_df["Correct"].sum()
    accuracy = correct / total
    
    logger.info(f"Predictions Summary: {correct}/{total} correct ({accuracy:.4f})")
    
    print(f"\n--- Predictions Summary ---")
    print(f"Total Predictions: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.4f}")
    
    if save:
        output_path = get_project_root() / "outputs" / "predictions.csv"
        save_dataframe(predictions_df, output_path)
    
    return predictions_df


def load_saved_model(
    model_filename: str = "fraud_detection_model.pkl"
) -> Any:
    """
    Load a saved model from disk.
    
    Args:
        model_filename: Filename of the saved model
        
    Returns:
        Loaded model instance
    """
    model_path = get_project_root() / "models" / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    return model


def load_saved_scaler(
    scaler_filename: str = "scaler.pkl"
) -> Any:
    """
    Load a saved scaler from disk.
    
    Args:
        scaler_filename: Filename of the saved scaler
        
    Returns:
        Loaded scaler instance
    """
    scaler_path = get_project_root() / "models" / scaler_filename
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    logger.info(f"Scaler loaded from {scaler_path}")
    
    return scaler
