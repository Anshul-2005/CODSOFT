#!/usr/bin/env python3
"""
Movie Rating Prediction — Main Pipeline
========================================

CodSoft Data Science Internship — Task 2

This script executes the complete ML pipeline:
    1. Load the IMDb India Movies dataset
    2. Explore and visualize the data (EDA)
    3. Clean and preprocess
    4. Engineer new features
    5. Train 6 regression models and compare
    6. Tune hyperparameters on top models
    7. Evaluate the best model comprehensively
    8. Save the best model to disk
    9. Generate predictions CSV
    10. Produce all output plots

Usage:
    python main.py

Requirements:
    - Place 'IMDb_Movies_India.csv' in the 'dataset/' folder
    - Install dependencies: pip install -r requirements.txt

Author: CodSoft Data Science Intern
"""

import sys
import warnings
import logging
from pathlib import Path

# ============================================================================
# PATH SETUP
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import numpy as np

from src.utils import (
    setup_directories,
    setup_logging,
    load_dataset,
    dataset_overview,
    print_section,
    OUTPUTS_DIR,
)
from src.preprocessing import (
    full_preprocessing_pipeline,
    scale_features,
    apply_scaler,
)
from src.feature_engineering import (
    full_feature_engineering,
    get_feature_columns,
)
from src.model_training import (
    split_data,
    train_and_compare,
    hyperparameter_tuning,
    save_model,
    save_scaler,
)
from src.evaluation import (
    full_evaluation,
    plot_prediction_vs_actual,
    plot_residuals,
    plot_feature_importance,
    generate_eda_plots,
)
from src.prediction import generate_predictions_csv, demo_predictions


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> int:
    """
    Execute the complete ML pipeline.
    
    Returns:
        0 on success, 1 on failure
    """
    logger = setup_logging()
    
    try:
        # ─────────────────────────────────────────────────────────────────────
        # STEP 0: Setup
        # ─────────────────────────────────────────────────────────────────────
        print_section("MOVIE RATING PREDICTION")
        print("  CodSoft Data Science Internship — Task 2")
        print("  " + "─" * 45)
        
        setup_directories()
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Load Dataset
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 1: LOAD DATASET")
        df_raw = load_dataset()
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Dataset Exploration
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 2: DATASET EXPLORATION")
        overview = dataset_overview(df_raw)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: EDA Visualizations (on raw data)
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 3: EDA VISUALIZATIONS")
        generate_eda_plots(df_raw)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Data Preprocessing
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 4: DATA PREPROCESSING")
        df = full_preprocessing_pipeline(df_raw.copy())
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Feature Engineering
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 5: FEATURE ENGINEERING")
        df = full_feature_engineering(df)
        
        # Display final feature set
        feature_cols = get_feature_columns(df)
        print(f"\n  Final Feature Set ({len(feature_cols)} features):")
        for i, col in enumerate(feature_cols, 1):
            print(f"    {i:2d}. {col}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Train-Test Split
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 6: TRAIN-TEST SPLIT")
        X_train, X_test, y_train, y_test = split_data(df)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 7: Feature Scaling
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 7: FEATURE SCALING")
        X_train_scaled, scaler = scale_features(X_train)
        X_test_scaled = apply_scaler(X_test, scaler)
        
        save_scaler(scaler)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 8: Model Training & Comparison
        # ─────────────────────────────────────────────────────────────────────
        comparison_df = train_and_compare(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        print("\n  Model Comparison Table (sorted by CV R²):")
        print(comparison_df.to_string(index=False))
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 9: Hyperparameter Tuning
        # ─────────────────────────────────────────────────────────────────────
        best_model, best_name, best_params, best_cv = hyperparameter_tuning(
            X_train_scaled, y_train, comparison_df
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 10: Final Evaluation
        # ─────────────────────────────────────────────────────────────────────
        metrics = full_evaluation(
            best_model, X_test_scaled, y_test,
            X_train_scaled, y_train, best_name
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 11: Evaluation Plots
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 11: EVALUATION PLOTS")
        plot_prediction_vs_actual(best_model, X_test_scaled, y_test)
        plot_residuals(best_model, X_test_scaled, y_test)
        plot_feature_importance(best_model, list(X_train.columns))
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 12: Save Model
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 12: SAVE MODEL")
        save_model(best_model)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 13: Generate Predictions
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 13: PREDICTIONS")
        generate_predictions_csv(best_model, X_test_scaled, y_test)
        demo_predictions(best_model, list(X_train.columns))
        
        # ─────────────────────────────────────────────────────────────────────
        # COMPLETE
        # ─────────────────────────────────────────────────────────────────────
        print_section("PIPELINE COMPLETE")
        print(f"""
  ✓ Dataset loaded and explored
  ✓ EDA visualizations generated
  ✓ Data preprocessed and features engineered
  ✓ 6 regression models trained and compared
  ✓ Hyperparameters tuned via GridSearchCV
  ✓ Best model: {best_name} (CV R²: {best_cv:.4f})
  ✓ Model saved to: models/movie_rating_model.pkl
  ✓ Scaler saved to: models/scaler.pkl
  ✓ Predictions saved to: outputs/predictions.csv
  ✓ All plots saved to: outputs/

  The project is ready for submission!
        """)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("  Please check the logs for more details.")
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
