"""
Model Training Module
======================

Handles training multiple models, comparison, and hyperparameter
tuning for the Credit Card Fraud Detection project.

Models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Gradient Boosting

Functions:
    - get_models: Return dictionary of models
    - train_single_model: Train one model
    - train_all_models: Train and compare all models
    - tune_best_model: Hyperparameter tuning with GridSearchCV
    - save_model: Save trained model to disk
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib

from src.utils import get_project_root, save_dataframe

# Logger
logger = logging.getLogger("CreditCardFraudDetection")


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.
    
    Returns:
        Dict: Model name to model instance mapping
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            max_depth=15
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            learning_rate=0.1
        ),
    }
    
    logger.info(f"Initialized {len(models)} models for training")
    return models


def train_single_model(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Train a single model and compute evaluation metrics.
    
    Args:
        model: Scikit-learn model instance
        model_name: Name of the model
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        Dict containing model, predictions, and metrics
    """
    logger.info(f"Training {model_name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probability predictions (for ROC-AUC)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    
    # Metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Training_Time_Sec": round(train_time, 2),
    }
    
    logger.info(
        f"{model_name} | Accuracy: {metrics['Accuracy']:.4f} | "
        f"Precision: {metrics['Precision']:.4f} | "
        f"Recall: {metrics['Recall']:.4f} | "
        f"F1: {metrics['F1_Score']:.4f} | "
        f"ROC-AUC: {metrics['ROC_AUC']:.4f} | "
        f"Time: {metrics['Training_Time_Sec']}s"
    )
    
    return {
        "model": model,
        "predictions": y_pred,
        "probabilities": y_prob,
        "metrics": metrics,
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train all models and create comparison DataFrame.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        Tuple of (results dict, comparison DataFrame)
    """
    logger.info("=" * 50)
    logger.info("Starting model training pipeline...")
    logger.info("=" * 50)
    
    models = get_models()
    results = {}
    comparison_data = []
    
    for name, model in models.items():
        result = train_single_model(
            model, name, X_train, y_train, X_test, y_test
        )
        results[name] = result
        comparison_data.append(result["metrics"])
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(by="F1_Score", ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)
    
    # Save comparison
    output_path = get_project_root() / "outputs" / "model_comparison.csv"
    save_dataframe(comparison_df, output_path)
    
    # Print comparison
    print("\n--- Model Comparison ---")
    print(comparison_df.to_string(index=False))
    
    logger.info("All models trained and compared successfully!")
    
    return results, comparison_df


def tune_best_model(
    best_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning on the best model using GridSearchCV.
    
    Args:
        best_model_name: Name of the best model to tune
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        
    Returns:
        Tuple of (best tuned model, best parameters)
    """
    logger.info(f"Tuning {best_model_name} with GridSearchCV...")
    
    # Define parameter grids for each model
    param_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "Decision Tree": {
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 15, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        },
    }
    
    # Get model and param grid
    models = get_models()
    model = models[best_model_name]
    param_grid = param_grids.get(best_model_name, {})
    
    if not param_grid:
        logger.warning(f"No param grid for {best_model_name}, returning base model")
        model.fit(X_train, y_train)
        return model, {}
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best CV F1 Score: {best_score:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        test_metrics["ROC_AUC"] = roc_auc_score(y_test, y_prob)
    
    print(f"\n--- Tuned {best_model_name} Results ---")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  Best Params: {best_params}")
    
    return best_model, best_params


def save_model(model: Any, filename: str = "fraud_detection_model.pkl") -> Path:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model instance
        filename: Output filename
        
    Returns:
        Path: Path to saved model
    """
    model_path = get_project_root() / "models" / filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path
