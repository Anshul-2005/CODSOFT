"""
Utility functions for Movie Rating Prediction project.
Handles directory setup, dataset loading, logging, and common helpers.

Author: CodSoft Data Science Intern
Task: Task 2 - Movie Rating Prediction
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

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
    logger = logging.getLogger("MovieRatingPrediction")
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def setup_directories() -> None:
    """
    Create all required project directories if they don't exist.
    """
    directories = [DATASET_DIR, MODELS_DIR, OUTPUTS_DIR, NOTEBOOK_DIR, SCREENSHOTS_DIR]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied creating directory: {directory}")
            raise
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise
    
    logger.info("Project directories verified/created successfully")


def verify_directory(path: Path, create: bool = True) -> bool:
    """
    Verify a directory exists, optionally creating it.
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

def load_dataset(filename: str = "IMDb_Movies_India.csv") -> pd.DataFrame:
    """
    Load the IMDb India Movies dataset.
    
    Args:
        filename: Name of the CSV file
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If dataset file is not found
    """
    filepath = DATASET_DIR / filename
    
    if not filepath.exists():
        # Try alternative common names
        alt_names = ["IMDb Movies India.csv", "imdb_india_movies.csv", "movies.csv"]
        for alt in alt_names:
            alt_path = DATASET_DIR / alt
            if alt_path.exists():
                filepath = alt_path
                break
        else:
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Dataset not found!\n"
                f"{'='*60}\n"
                f"Expected location: {DATASET_DIR / filename}\n\n"
                f"Please download the dataset from:\n"
                f"  https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies\n\n"
                f"Then place the CSV file in the 'dataset/' folder.\n"
                f"See 'dataset/download_dataset.md' for detailed instructions.\n"
                f"{'='*60}"
            )
            logger.error(f"Dataset not found at: {filepath}")
            raise FileNotFoundError(error_msg)
    
    if not os.access(filepath, os.R_OK):
        logger.error(f"No read permission for: {filepath}")
        raise PermissionError(f"Cannot read dataset file: {filepath}")
    
    # Try multiple encodings
    encodings = ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns (encoding: {encoding})")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"The dataset file is empty: {filepath}")
        except Exception as e:
            logger.warning(f"Failed with encoding {encoding}: {e}")
            continue
    
    raise ValueError(f"Could not read CSV with any standard encoding: {filepath}")


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def save_plot(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """
    Save a matplotlib figure to the outputs directory.
    """
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
    """
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
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# ============================================================================
# DATASET OVERVIEW
# ============================================================================

def dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Print and return comprehensive dataset overview.
    """
    print_section("DATASET OVERVIEW")
    
    overview = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
    }
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    
    print_subsection("Data Types")
    print(df.dtypes.to_string())
    
    print_subsection("Missing Values")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Count": missing, "Percent (%)": missing_pct})
    missing_df = missing_df[missing_df["Count"] > 0].sort_values("Percent (%)", ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  No missing values found.")
    
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    # Target distribution (Rating)
    if "Rating" in df.columns:
        print_subsection("Target Distribution (Rating)")
        rating_stats = df["Rating"].describe()
        print(rating_stats.to_string())
        overview["rating_mean"] = df["Rating"].mean()
        overview["rating_std"] = df["Rating"].std()
    
    print_subsection("Sample Data (First 5 Rows)")
    print(df.head().to_string())
    
    return overview


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that a DataFrame has required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.warning(f"Missing required columns: {missing}")
        return False
    return True
