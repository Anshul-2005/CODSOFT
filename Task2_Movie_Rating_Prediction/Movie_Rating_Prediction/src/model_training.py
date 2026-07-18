"""
Model training module for Movie Rating Prediction.
Trains multiple regressors, compares performance, and performs hyperparameter tuning.

Author: CodSoft Data Science Intern
Task: Task 2 - Movie Rating Prediction
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
    KFold,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.utils import MODELS_DIR, OUTPUTS_DIR, print_section, verify_directory, save_dataframe

logger = logging.getLogger("MovieRatingPrediction")


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data(
    df: pd.DataFrame,
    target: str = "Rating",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.
    
    Args:
        df: DataFrame containing features and target
        target: Name of the target column
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"Data split: Train={len(X_train)} ({1-test_size:.0%}), Test={len(X_test)} ({test_size:.0%})")
    logger.info(f"  Train target mean: {y_train.mean():.2f}")
    logger.info(f"  Test target mean: {y_test.mean():.2f}")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models() -> Dict[str, Any]:
    """
    Return dictionary of regression models to train.
    
    Models:
        1. Linear Regression: Simple linear baseline
        2. Ridge Regression: L2 regularized linear model
        3. Lasso Regression: L1 regularized linear model
        4. Decision Tree: Non-linear, interpretable
        5. Random Forest: Ensemble, reduces overfitting
        6. Gradient Boosting: Sequential ensemble
    
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
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
        "Linear Regression": {},  # No hyperparameters
        "Ridge Regression": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "Lasso Regression": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, 15, None],
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
    
    Metrics:
        - MAE: Mean Absolute Error
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - R²: Coefficient of Determination
        - CV Mean: Cross-validation mean R²
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        cv_folds: Number of cross-validation folds
    
    Returns:
        DataFrame with model comparison results
    """
    print_section("MODEL TRAINING & COMPARISON")
    
    models = get_models()
    results = []
    
    header = f"{'Model':25s} | {'MAE':>8s} | {'RMSE':>8s} | {'R²':>8s} | {'CV R²':>10s}"
    print(f"\n{header}")
    print("-" * len(header))
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R² (Train)": round(r2_train, 4),
            "R² (Test)": round(r2_test, 4),
            "CV R² Mean": round(cv_mean, 4),
            "CV R² Std": round(cv_std, 4),
        })
        
        print(f"{name:25s} | {mae:>8.4f} | {rmse:>8.4f} | {r2_test:>8.4f} | {cv_mean:>6.4f}±{cv_std:.4f}")
    
    # Create comparison DataFrame sorted by CV R²
    comparison_df = pd.DataFrame(results).sort_values(
        "CV R² Mean", ascending=False
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
    
    # Get top models (exclude Linear Regression as it has no hyperparameters)
    top_models = comparison_df.head(top_n + 1)["Model"].tolist()
    top_models = [m for m in top_models if m != "Linear Regression"][:top_n]
    
    logger.info(f"Tuning top {len(top_models)} models: {top_models}")
    
    param_grids = get_param_grids()
    all_models = get_models()
    
    best_model = None
    best_score = -float('inf')
    best_name = ""
    best_params = {}
    
    for model_name in top_models:
        if model_name not in param_grids or not param_grids[model_name]:
            logger.warning(f"No param grid for {model_name}, skipping")
            continue
        
        print(f"\nTuning: {model_name}...")
        
        base_model = all_models[model_name]
        param_grid = param_grids[model_name]
        
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        
        grid.fit(X_train, y_train)
        
        print(f"  Best Parameters: {grid.best_params_}")
        print(f"  Best CV R²:      {grid.best_score_:.4f}")
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = model_name
            best_params = grid.best_params_
    
    print(f"\n{'='*50}")
    print(f"FINAL BEST MODEL: {best_name}")
    print(f"BEST CV R²: {best_score:.4f}")
    print(f"BEST PARAMETERS: {best_params}")
    print(f"{'='*50}")
    
    logger.info(f"Hyperparameter tuning complete. Best: {best_name} (R²: {best_score:.4f})")
    
    return best_model, best_name, best_params, best_score


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model: Any, filename: str = "movie_rating_model.pkl") -> Path:
    """
    Save trained model to disk using joblib.
    """
    verify_directory(MODELS_DIR)
    filepath = MODELS_DIR / filename
    
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise IOError(f"Failed to save model to {filepath}: {e}")


def load_model(filename: str = "movie_rating_model.pkl") -> Any:
    """
    Load a trained model from disk.
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
    """Save fitted scaler to disk."""
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
    """Load a fitted scaler from disk."""
    filepath = MODELS_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found at: {filepath}")
    
    return joblib.load(filepath)
