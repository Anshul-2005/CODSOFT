"""
Model training module for Titanic Survival Prediction.
Trains multiple classifiers, compares performance, and performs hyperparameter tuning.

Author: CodSoft Data Science Intern
Task: Task 1 - Titanic Survival Prediction
"""

from typing import Tuple, Dict, List, Any, Optional
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

from src.utils import MODELS_DIR, OUTPUTS_DIR, print_section, verify_directory, save_dataframe

# Get logger
logger = logging.getLogger("TitanicPrediction")


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data(
    df: pd.DataFrame,
    target: str = "Survived",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets with stratification.
    
    Args:
        df: DataFrame containing features and target
        target: Name of the target column
        test_size: Proportion for test set (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    
    Raises:
        ValueError: If target column not found
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # Maintain class distribution
    )
    
    logger.info(f"Data split: Train={len(X_train)} ({1-test_size:.0%}), Test={len(X_test)} ({test_size:.0%})")
    logger.info(f"  Train survival rate: {y_train.mean():.2%}")
    logger.info(f"  Test survival rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models() -> Dict[str, Any]:
    """
    Return dictionary of classification models to train.
    
    Models:
        1. Logistic Regression: Linear baseline
        2. Decision Tree: Non-linear, interpretable
        3. Random Forest: Ensemble, reduces overfitting
        4. Gradient Boosting: Sequential ensemble
        5. KNN: Instance-based learning
        6. SVM: Margin-based classifier
    
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            random_state=42,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", 
            probability=True, 
            random_state=42
        ),
    }
    return models


def get_param_grids() -> Dict[str, Dict[str, List]]:
    """
    Return hyperparameter grids for GridSearchCV.
    
    Returns:
        Dictionary mapping model names to parameter grids
    """
    return {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    }


# ============================================================================
# MODEL TRAINING & COMPARISON
# ============================================================================

def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Train all models, evaluate with cross-validation, and create comparison table.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        cv_folds: Number of cross-validation folds
    
    Returns:
        DataFrame with model comparison results, sorted by CV Mean
    """
    print_section("MODEL TRAINING & COMPARISON")
    
    models = get_models()
    results = []
    trained_models = {}
    
    header = f"{'Model':25s} | {'Train':>10s} | {'Test':>10s} | {'CV Mean':>10s} | {'CV Std':>10s}"
    print(f"\n{header}")
    print("-" * len(header))
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Cross-validation with stratification
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            "Model": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "CV Mean": round(cv_mean, 4),
            "CV Std": round(cv_std, 4),
        })
        
        print(f"{name:25s} | {train_acc:>10.4f} | {test_acc:>10.4f} | {cv_mean:>10.4f} | {cv_std:>10.4f}")
    
    # Create comparison DataFrame sorted by CV Mean
    comparison_df = pd.DataFrame(results).sort_values(
        "CV Mean", ascending=False
    ).reset_index(drop=True)
    
    # Save comparison to outputs
    verify_directory(OUTPUTS_DIR)
    save_dataframe(comparison_df, "model_comparison.csv")
    
    logger.info(f"Model comparison complete. Best: {comparison_df.iloc[0]['Model']}")
    
    return comparison_df


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    comparison_df: pd.DataFrame,
    top_n: int = 2,
    cv_folds: int = 5,
) -> Tuple[Any, str, Dict, float]:
    """
    Perform GridSearchCV on the top performing models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        comparison_df: Model comparison DataFrame
        top_n: Number of top models to tune
        cv_folds: Number of cross-validation folds
    
    Returns:
        Tuple of (best_model, best_model_name, best_params, best_cv_score)
    """
    print_section("HYPERPARAMETER TUNING")
    
    # Get top models
    top_models = comparison_df.head(top_n)["Model"].tolist()
    logger.info(f"Tuning top {top_n} models: {top_models}")
    
    param_grids = get_param_grids()
    all_models = get_models()
    
    best_model = None
    best_score = 0
    best_name = ""
    best_params = {}
    
    for model_name in top_models:
        if model_name not in param_grids:
            logger.warning(f"No param grid for {model_name}, skipping")
            continue
        
        print(f"\nTuning: {model_name}...")
        
        # Get fresh model instance and param grid
        base_model = all_models[model_name]
        param_grid = param_grids[model_name]
        
        # GridSearchCV with stratified CV
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        
        grid.fit(X_train, y_train)
        
        print(f"  Best Parameters: {grid.best_params_}")
        print(f"  Best CV Score:   {grid.best_score_:.4f}")
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = model_name
            best_params = grid.best_params_
    
    print(f"\n{'='*50}")
    print(f"FINAL BEST MODEL: {best_name}")
    print(f"BEST CV SCORE: {best_score:.4f}")
    print(f"BEST PARAMETERS: {best_params}")
    print(f"{'='*50}")
    
    logger.info(f"Hyperparameter tuning complete. Best: {best_name} (CV: {best_score:.4f})")
    
    return best_model, best_name, best_params, best_score


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model: Any, filename: str = "titanic_model.pkl") -> Path:
    """
    Save trained model to disk using joblib.
    
    Args:
        model: Trained model to save
        filename: Output filename
    
    Returns:
        Path to saved model file
    
    Raises:
        IOError: If save fails
    """
    # Ensure models directory exists
    verify_directory(MODELS_DIR)
    
    filepath = MODELS_DIR / filename
    
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise IOError(f"Failed to save model to {filepath}: {e}")


def load_model(filename: str = "titanic_model.pkl") -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filename: Model filename
    
    Returns:
        Loaded model
    
    Raises:
        FileNotFoundError: If model file not found
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
        logger.info(f"Model loaded: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def save_scaler(scaler: Any, filename: str = "scaler.pkl") -> Path:
    """
    Save fitted scaler to disk.
    
    Args:
        scaler: Fitted scaler to save
        filename: Output filename
    
    Returns:
        Path to saved scaler file
    """
    verify_directory(MODELS_DIR)
    filepath = MODELS_DIR / filename
    
    try:
        joblib.dump(scaler, filepath)
        logger.info(f"Scaler saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save scaler: {e}")
        raise


def load_scaler(filename: str = "scaler.pkl") -> Any:
    """
    Load a fitted scaler from disk.
    
    Args:
        filename: Scaler filename
    
    Returns:
        Loaded scaler
    """
    filepath = MODELS_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found at: {filepath}")
    
    return joblib.load(filepath)
