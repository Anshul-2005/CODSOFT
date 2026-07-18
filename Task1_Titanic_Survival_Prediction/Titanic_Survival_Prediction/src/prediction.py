"""
Prediction module for Titanic Survival Prediction.
Provides inference capabilities for trained models.

Author: CodSoft Data Science Intern
Task: Task 1 - Titanic Survival Prediction
"""

from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from src.utils import MODELS_DIR, OUTPUTS_DIR, verify_directory, print_section

# Get logger
logger = logging.getLogger("TitanicPrediction")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(filename: str = "titanic_model.pkl") -> Any:
    """
    Load the saved trained model.
    
    Args:
        filename: Model filename
    
    Returns:
        Loaded model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
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

def predict_survival(
    model: Any,
    feature_names: List[str],
    passenger_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Predict survival for a single passenger.
    
    Args:
        model: Trained model
        feature_names: List of expected feature names (in order)
        passenger_data: Dictionary of passenger features
        **kwargs: Alternative way to pass passenger features
    
    Returns:
        Dictionary with prediction and probability:
            {
                "Prediction": "Survived" or "Not Survived",
                "Prediction_Code": 0 or 1,
                "Probability_Survived": "XX.XX%",
                "Probability_Not_Survived": "XX.XX%"
            }
    
    Raises:
        ValueError: If required features are missing
    """
    # Merge passenger_data and kwargs
    data = passenger_data or {}
    data.update(kwargs)
    
    if not data:
        raise ValueError("No passenger data provided")
    
    # Build feature vector, filling missing features with 0
    input_data = {}
    for feat in feature_names:
        if feat in data:
            input_data[feat] = data[feat]
        else:
            input_data[feat] = 0
            logger.debug(f"Missing feature '{feat}', using default value 0")
    
    # Create DataFrame
    X = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probability if available
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[0]
    
    result = {
        "Prediction": "Survived" if prediction == 1 else "Not Survived",
        "Prediction_Code": int(prediction),
        "Probability_Survived": f"{probability[1] * 100:.2f}%" if probability is not None else "N/A",
        "Probability_Not_Survived": f"{probability[0] * 100:.2f}%" if probability is not None else "N/A",
    }
    
    return result


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
        - Actual: True labels
        - Predicted: Model predictions
        - Probability_Survived: Survival probability (%)
        - Correct: 1 if prediction matches actual, 0 otherwise
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        filename: Output filename
    
    Returns:
        Path to saved CSV file
    """
    verify_directory(OUTPUTS_DIR)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = np.full(len(y_pred), np.nan)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Probability_Survived": np.round(y_proba * 100, 2),
        "Correct": (y_test.values == y_pred).astype(int),
    })
    
    # Save to CSV
    filepath = OUTPUTS_DIR / filename
    predictions_df.to_csv(filepath, index=False)
    
    # Log summary
    correct = predictions_df["Correct"].sum()
    total = len(predictions_df)
    accuracy = correct / total
    
    logger.info(f"Predictions saved: {filepath}")
    logger.info(f"  Correct: {correct}/{total} ({accuracy:.2%})")
    
    return filepath


# ============================================================================
# DEMO PREDICTIONS
# ============================================================================

def demo_predictions(model: Any, feature_names: List[str]) -> None:
    """
    Run sample predictions to demonstrate the model.
    
    Demonstrates predictions for three example passengers:
        1. Upper-class woman (high survival probability)
        2. Third-class man (low survival probability)
        3. Child (moderate survival probability)
    
    Args:
        model: Trained model
        feature_names: List of expected feature names
    """
    print_section("SAMPLE PREDICTIONS")
    
    # Example 1: Upper-class woman
    passenger_1 = {
        "Pclass": 1,
        "Sex": 0,  # female
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 211.3,
        "FamilySize": 1,
        "IsAlone": 1,
        "CabinAvailable": 1,
        "TicketFrequency": 1,
        "Title": 2,  # Mrs
        "Embarked_C": 0,
        "Embarked_Q": 0,
        "Embarked_S": 1,
        "FareCategory_High": 0,
        "FareCategory_Low": 0,
        "FareCategory_Medium": 0,
        "FareCategory_VeryHigh": 1,
        "AgeCategory_Adult": 1,
        "AgeCategory_Child": 0,
        "AgeCategory_Senior": 0,
        "AgeCategory_Teen": 0,
    }
    
    result1 = predict_survival(model, feature_names, passenger_1)
    print(f"\n  Passenger 1: 1st Class Female, Age 29, Fare $211.30")
    print(f"    Prediction:  {result1['Prediction']}")
    print(f"    Probability: {result1['Probability_Survived']}")
    
    # Example 2: Third-class man
    passenger_2 = {
        "Pclass": 3,
        "Sex": 1,  # male
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "FamilySize": 2,
        "IsAlone": 0,
        "CabinAvailable": 0,
        "TicketFrequency": 1,
        "Title": 0,  # Mr
        "Embarked_C": 0,
        "Embarked_Q": 0,
        "Embarked_S": 1,
        "FareCategory_High": 0,
        "FareCategory_Low": 1,
        "FareCategory_Medium": 0,
        "FareCategory_VeryHigh": 0,
        "AgeCategory_Adult": 1,
        "AgeCategory_Child": 0,
        "AgeCategory_Senior": 0,
        "AgeCategory_Teen": 0,
    }
    
    result2 = predict_survival(model, feature_names, passenger_2)
    print(f"\n  Passenger 2: 3rd Class Male, Age 22, Fare $7.25")
    print(f"    Prediction:  {result2['Prediction']}")
    print(f"    Probability: {result2['Probability_Survived']}")
    
    # Example 3: Child
    passenger_3 = {
        "Pclass": 2,
        "Sex": 1,  # male
        "Age": 5,
        "SibSp": 1,
        "Parch": 2,
        "Fare": 21.0,
        "FamilySize": 4,
        "IsAlone": 0,
        "CabinAvailable": 0,
        "TicketFrequency": 3,
        "Title": 3,  # Master
        "Embarked_C": 0,
        "Embarked_Q": 0,
        "Embarked_S": 1,
        "FareCategory_High": 0,
        "FareCategory_Low": 0,
        "FareCategory_Medium": 1,
        "FareCategory_VeryHigh": 0,
        "AgeCategory_Adult": 0,
        "AgeCategory_Child": 1,
        "AgeCategory_Senior": 0,
        "AgeCategory_Teen": 0,
    }
    
    result3 = predict_survival(model, feature_names, passenger_3)
    print(f"\n  Passenger 3: 2nd Class Male Child, Age 5, Fare $21.00")
    print(f"    Prediction:  {result3['Prediction']}")
    print(f"    Probability: {result3['Probability_Survived']}")


# ============================================================================
# PREDICTION UTILITIES
# ============================================================================

def validate_input(
    data: Dict[str, Any], 
    required_fields: List[str],
) -> bool:
    """
    Validate prediction input data.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
    
    Returns:
        True if valid, False otherwise
    """
    missing = set(required_fields) - set(data.keys())
    if missing:
        logger.warning(f"Missing required fields: {missing}")
        return False
    return True


def preprocess_single_input(
    pclass: int,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: str,
    cabin: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Preprocess raw passenger input into model-ready features.
    
    This function handles the feature engineering for a single passenger
    to make predictions without needing the full preprocessing pipeline.
    
    Args:
        pclass: Passenger class (1, 2, or 3)
        sex: Gender ('male' or 'female')
        age: Age in years
        sibsp: Number of siblings/spouses aboard
        parch: Number of parents/children aboard
        fare: Passenger fare
        embarked: Port of embarkation ('C', 'Q', or 'S')
        cabin: Cabin number (optional)
        name: Full name for title extraction (optional)
    
    Returns:
        Dictionary of processed features
    """
    # Basic features
    features = {
        "Pclass": pclass,
        "Sex": 1 if sex.lower() == "male" else 0,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
    }
    
    # Engineered features
    features["FamilySize"] = sibsp + parch + 1
    features["IsAlone"] = 1 if features["FamilySize"] == 1 else 0
    features["CabinAvailable"] = 1 if cabin and cabin.lower() != "unknown" else 0
    features["TicketFrequency"] = 1  # Default for single prediction
    
    # Title extraction (simplified)
    if name:
        if "Mr." in name:
            features["Title"] = 0
        elif "Miss." in name or "Ms." in name:
            features["Title"] = 1
        elif "Mrs." in name:
            features["Title"] = 2
        elif "Master." in name:
            features["Title"] = 3
        else:
            features["Title"] = 4  # Rare
    else:
        # Infer from sex and age
        if sex.lower() == "male":
            features["Title"] = 3 if age < 15 else 0  # Master or Mr
        else:
            features["Title"] = 1  # Miss (default for female)
    
    # Embarked one-hot
    features["Embarked_C"] = 1 if embarked.upper() == "C" else 0
    features["Embarked_Q"] = 1 if embarked.upper() == "Q" else 0
    features["Embarked_S"] = 1 if embarked.upper() == "S" else 0
    
    # Fare category
    if fare < 7.91:
        features["FareCategory_Low"] = 1
    elif fare < 14.45:
        features["FareCategory_Medium"] = 1
    elif fare < 31.0:
        features["FareCategory_High"] = 1
    else:
        features["FareCategory_VeryHigh"] = 1
    
    # Set defaults for fare categories not matched
    for cat in ["FareCategory_Low", "FareCategory_Medium", "FareCategory_High", "FareCategory_VeryHigh"]:
        if cat not in features:
            features[cat] = 0
    
    # Age category
    if age <= 12:
        features["AgeCategory_Child"] = 1
    elif age <= 18:
        features["AgeCategory_Teen"] = 1
    elif age <= 60:
        features["AgeCategory_Adult"] = 1
    else:
        features["AgeCategory_Senior"] = 1
    
    # Set defaults for age categories not matched
    for cat in ["AgeCategory_Child", "AgeCategory_Teen", "AgeCategory_Adult", "AgeCategory_Senior"]:
        if cat not in features:
            features[cat] = 0
    
    return features
