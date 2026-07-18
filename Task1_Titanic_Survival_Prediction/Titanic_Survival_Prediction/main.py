#!/usr/bin/env python3
"""
Titanic Survival Prediction — Main Pipeline
============================================

CodSoft Data Science Internship — Task 1

This script executes the complete ML pipeline:
    1. Load the Titanic dataset
    2. Explore and visualize the data (EDA)
    3. Clean and preprocess
    4. Engineer new features
    5. Train 6 ML models and compare
    6. Tune hyperparameters on top models
    7. Evaluate the best model comprehensively
    8. Save the best model to disk
    9. Generate predictions CSV
    10. Produce all output plots

Usage:
    python main.py

Requirements:
    - Place 'Titanic-Dataset.csv' in the 'dataset/' folder
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

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# IMPORTS (after path setup)
# ============================================================================

import pandas as pd

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
    encode_categorical,
    drop_unnecessary_columns,
    scale_features,
    apply_scaler,
)
from src.feature_engineering import full_feature_engineering
from src.model_training import (
    split_data,
    train_and_compare,
    hyperparameter_tuning,
    save_model,
    save_scaler,
)
from src.evaluation import (
    full_evaluation,
    plot_confusion_matrix,
    plot_roc_curve,
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
    # Initialize logging
    logger = setup_logging()
    
    try:
        # ─────────────────────────────────────────────────────────────────────
        # STEP 0: Setup
        # ─────────────────────────────────────────────────────────────────────
        print_section("TITANIC SURVIVAL PREDICTION")
        print("  CodSoft Data Science Internship — Task 1")
        print("  " + "─" * 45)
        
        setup_directories()
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Load Dataset
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 1: LOAD DATASET")
        df_raw = load_dataset("Titanic-Dataset.csv")
        
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
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Encoding & Column Dropping
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 6: ENCODING & CLEANUP")
        df = encode_categorical(df)
        df = drop_unnecessary_columns(df)
        
        # Display final feature set
        feature_cols = [c for c in df.columns if c != "Survived"]
        print(f"\n  Final Feature Set ({len(feature_cols)} features):")
        for i, col in enumerate(feature_cols, 1):
            print(f"    {i:2d}. {col}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 7: Train-Test Split
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 7: TRAIN-TEST SPLIT")
        X_train, X_test, y_train, y_test = split_data(df)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 8: Feature Scaling
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 8: FEATURE SCALING")
        X_train_scaled, scaler = scale_features(X_train)
        X_test_scaled = apply_scaler(X_test, scaler)
        
        # Save scaler for future use
        save_scaler(scaler)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 9: Model Training & Comparison
        # ─────────────────────────────────────────────────────────────────────
        comparison_df = train_and_compare(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        print("\n  Model Comparison Table (sorted by CV Mean):")
        print(comparison_df.to_string(index=False))
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 10: Hyperparameter Tuning
        # ─────────────────────────────────────────────────────────────────────
        best_model, best_name, best_params, best_cv = hyperparameter_tuning(
            X_train_scaled, y_train, comparison_df
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 11: Final Evaluation
        # ─────────────────────────────────────────────────────────────────────
        metrics = full_evaluation(
            best_model, X_test_scaled, y_test, 
            X_train_scaled, y_train, best_name
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 12: Evaluation Plots
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 12: EVALUATION PLOTS")
        plot_confusion_matrix(best_model, X_test_scaled, y_test)
        plot_roc_curve(best_model, X_test_scaled, y_test)
        plot_feature_importance(best_model, list(X_train.columns))
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 13: Save Model
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 13: SAVE MODEL")
        save_model(best_model)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 14: Generate Predictions
        # ─────────────────────────────────────────────────────────────────────
        print_section("STEP 14: PREDICTIONS")
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
  ✓ 6 models trained and compared
  ✓ Hyperparameters tuned via GridSearchCV
  ✓ Best model: {best_name} (CV: {best_cv:.4f})
  ✓ Model saved to: models/titanic_model.pkl
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
