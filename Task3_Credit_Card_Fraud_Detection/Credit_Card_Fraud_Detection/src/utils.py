"""
Utility Functions
=================

Helper functions used across the Credit Card Fraud Detection project.

Functions:
    - setup_logging: Configure logging for the project
    - create_directories: Create required project directories
    - get_project_root: Get the project root path
    - save_dataframe: Save DataFrame to CSV
    - load_dataframe: Load DataFrame from CSV
    - set_plot_style: Set consistent matplotlib style
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory path
    """
    return Path(__file__).parent.parent


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        log_level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("CreditCardFraudDetection")
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def create_directories() -> None:
    """
    Create all required project directories if they don't exist.
    """
    project_root = get_project_root()
    
    directories = [
        project_root / "dataset",
        project_root / "notebook",
        project_root / "src",
        project_root / "models",
        project_root / "outputs",
        project_root / "screenshots",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep files
    for gitkeep_dir in ["models", "outputs"]:
        gitkeep_path = project_root / gitkeep_dir / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()


def save_dataframe(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        index: Whether to include index (default: False)
    """
    logger = logging.getLogger("CreditCardFraudDetection")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    logger.info(f"DataFrame saved to {filepath}")


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Input file path
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    logger = logging.getLogger("CreditCardFraudDetection")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"DataFrame loaded from {filepath} | Shape: {df.shape}")
    return df


def set_plot_style() -> None:
    """
    Set consistent matplotlib/seaborn style for all visualizations.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })


def print_section_header(title: str) -> None:
    """
    Print a formatted section header for console output.
    
    Args:
        title: Section title
    """
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
