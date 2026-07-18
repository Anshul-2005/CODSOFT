"""
Utility functions for Titanic Survival Prediction project.
Handles directory setup, dataset loading, logging, and common helpers.

Author: CodSoft Data Science Intern
Task: Task 1 - Titanic Survival Prediction
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# PROJECT PATHS (Platform-Independent)
# ============================================================================

def get_project_root() -> Path:
    """
    Determine the project root directory.
    Works regardless of where the script is executed from.
    """
    # If running from src/, go up one level
    current_file = Path(__file__).resolve()
    if current_file.parent.name == "src":
        return current_file.parent.parent
    return current_file.parent


PROJECT_ROOT = get_project_root()
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("TitanicPrediction")
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Initialize logger
logger = setup_logging()


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def setup_directories() -> None:
    """
    Create all required project directories if they don't exist.
    
    Creates:
        - dataset/
        - models/
        - outputs/
        - notebook/
        - screenshots/
    """
    directories = [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, NOTEBOOK_DIR, SCREENSHOTS_DIR]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Permission denied creating directory: {directory}")
            raise
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise
    
    logger.info("Project directories verified/created successfully")


def verify_directory(path: Path, create: bool = True) -> bool:
    """
    Verify a directory exists, optionally creating it.
    
    Args:
        path: Directory path to verify
        create: Whether to create if missing
    
    Returns:
        True if directory exists/was created, False otherwise
    """
    if path.exists():
        return True
    
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    return False


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_dataset(filename: str = "Titanic-Dataset.csv") -> pd.DataFrame:
    """
    Load the Titanic dataset from the dataset directory.
    
    Args:
        filename: Name of the CSV file (default: Titanic-Dataset.csv)
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If dataset file is not found
        pd.errors.EmptyDataError: If CSV file is empty
        pd.errors.ParserError: If CSV file is corrupted
    """
    filepath = DATASET_DIR / filename
    
    # Check if file exists
    if not filepath.exists():
        error_msg = (
            f"\n{'='*60}\n"
            f"ERROR: Dataset not found!\n"
            f"{'='*60}\n"
            f"Expected location: {filepath}\n\n"
            f"Please download the dataset from:\n"
            f"  https://www.kaggle.com/datasets/yasserh/titanic-dataset\n\n"
            f"Then place 'Titanic-Dataset.csv' in the 'dataset/' folder.\n"
            f"See 'dataset/download_dataset.md' for detailed instructions.\n"
            f"{'='*60}"
        )
        logger.error(f"Dataset not found at: {filepath}")
        raise FileNotFoundError(error_msg)
    
    # Check if file is readable
    if not os.access(filepath, os.R_OK):
        logger.error(f"No read permission for: {filepath}")
        raise PermissionError(f"Cannot read dataset file: {filepath}")
    
    # Load the CSV
    try:
        df = pd.read_csv(filepath)
    except pd.errors.EmptyDataError:
        logger.error("Dataset file is empty")
        raise pd.errors.EmptyDataError(f"The dataset file is empty: {filepath}")
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise pd.errors.ParserError(f"CSV file is corrupted or invalid: {filepath}")
    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        raise
    
    # Validate expected columns
    expected_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
                        'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    missing_cols = set(expected_columns) - set(df.columns)
    
    if missing_cols:
        logger.warning(f"Missing expected columns: {missing_cols}")
    
    logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def save_plot(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """
    Save a matplotlib figure to the outputs directory.
    
    Args:
        fig: Matplotlib figure to save
        filename: Output filename (e.g., 'confusion_matrix.png')
        dpi: Resolution (default: 150)
    
    Returns:
        Path to saved file
    """
    # Ensure outputs directory exists
    verify_directory(OUTPUTS_DIR)
    
    filepath = OUTPUTS_DIR / filename
    
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved plot: {filename}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save plot {filename}: {e}")
        raise


def save_dataframe(df: pd.DataFrame, filename: str, index: bool = False) -> Path:
    """
    Save a DataFrame to the outputs directory as CSV.
    
    Args:
        df: DataFrame to save
        filename: Output filename (e.g., 'predictions.csv')
        index: Whether to include row index (default: False)
    
    Returns:
        Path to saved file
    """
    # Ensure outputs directory exists
    verify_directory(OUTPUTS_DIR)
    
    filepath = OUTPUTS_DIR / filename
    
    try:
        df.to_csv(filepath, index=index)
        logger.info(f"Saved CSV: {filename}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save CSV {filename}: {e}")
        raise


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def print_section(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title text
        width: Total width of the header line
    """
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# ============================================================================
# DATASET OVERVIEW
# ============================================================================

def dataset_overview(df: pd.DataFrame) -> dict:
    """
    Print and return comprehensive dataset overview.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary containing overview statistics
    """
    print_section("DATASET OVERVIEW")
    
    overview = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
    }
    
    # Shape
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Columns
    print(f"\nColumns: {list(df.columns)}")
    
    # Data Types
    print_subsection("Data Types")
    print(df.dtypes.to_string())
    
    # Missing Values
    print_subsection("Missing Values")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Count": missing, "Percent (%)": missing_pct})
    missing_df = missing_df[missing_df["Count"] > 0].sort_values("Percent (%)", ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  No missing values found.")
    
    # Duplicates
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    # Target Distribution
    if "Survived" in df.columns:
        print_subsection("Target Distribution (Survived)")
        print(df["Survived"].value_counts().to_string())
        print(f"  Survival Rate: {df['Survived'].mean():.2%}")
        overview["survival_rate"] = df["Survived"].mean()
    
    # Summary Statistics
    print_subsection("Summary Statistics (Numerical)")
    print(df.describe().round(2).to_string())
    
    # Unique Values
    print_subsection("Unique Values per Column")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()}")
    
    return overview


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if all columns present, False otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.warning(f"Missing required columns: {missing}")
        return False
    return True


def validate_model_input(X: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """
    Validate and align model input features.
    
    Args:
        X: Input DataFrame
        expected_features: List of expected feature names
    
    Returns:
        Aligned DataFrame with correct columns
    """
    # Add missing columns with zeros
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
            logger.warning(f"Added missing feature with default value: {col}")
    
    # Reorder columns to match expected order
    X = X[expected_features]
    
    return X
