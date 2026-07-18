"""
Model evaluation module for Movie Rating Prediction.
Generates metrics, plots, and evaluation reports.

Author: CodSoft Data Science Intern
Task: Task 2 - Movie Rating Prediction
"""

from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, KFold

from src.utils import OUTPUTS_DIR, save_plot, print_section, verify_directory

logger = logging.getLogger("MovieRatingPrediction")


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def full_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with all regression metrics.
    
    Metrics:
        - MAE: Mean Absolute Error
        - MSE: Mean Squared Error
        - RMSE: Root Mean Squared Error
        - R²: Coefficient of Determination
        - Cross-validation R²
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    print_section("MODEL EVALUATION")
    
    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    
    metrics = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R² (Train)": round(r2_train, 4),
        "R² (Test)": round(r2_test, 4),
        "CV R² Mean": round(cv_scores.mean(), 4),
        "CV R² Std": round(cv_scores.std(), 4),
    }
    
    # Print formatted results
    print(f"\n  Model: {model_name}")
    print(f"  {'─' * 50}")
    print(f"  MAE:       {mae:.4f}")
    print(f"    → Average absolute error in rating prediction")
    print(f"  MSE:       {mse:.4f}")
    print(f"    → Average squared error (penalizes large errors)")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"    → Root of MSE (same units as rating)")
    print(f"  R² Train:  {r2_train:.4f}")
    print(f"    → Variance explained on training data")
    print(f"  R² Test:   {r2_test:.4f}")
    print(f"    → Variance explained on test data")
    print(f"  CV R²:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    → Average R² across 5 folds")
    
    # Prediction statistics
    print(f"\n  Prediction Statistics:")
    print(f"    Actual range:    {y_test.min():.2f} - {y_test.max():.2f}")
    print(f"    Predicted range: {y_pred_test.min():.2f} - {y_pred_test.max():.2f}")
    print(f"    Mean residual:   {(y_test - y_pred_test).mean():.4f}")
    
    logger.info(f"Evaluation complete. R²: {r2_test:.4f}, RMSE: {rmse:.4f}")
    
    return metrics


# ============================================================================
# PREDICTION VS ACTUAL PLOT
# ============================================================================

def plot_prediction_vs_actual(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """
    Generate prediction vs actual scatter plot.
    
    The ideal model would have all points on the diagonal line.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save: Whether to save the plot
    
    Returns:
        Matplotlib figure
    """
    y_pred = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="white", linewidth=0.5, s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
    
    # R² annotation
    r2 = r2_score(y_test, y_pred)
    ax.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    ax.set_xlabel("Actual Rating", fontsize=13)
    ax.set_ylabel("Predicted Rating", fontsize=13)
    ax.set_title("Prediction vs Actual Rating", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, "prediction_vs_actual.png")
    
    return fig


# ============================================================================
# RESIDUAL PLOT
# ============================================================================

def plot_residuals(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """
    Generate residual analysis plots.
    
    Residuals = Actual - Predicted
    
    Good model should have:
        - Residuals centered around 0
        - No pattern in residuals vs predicted
        - Normally distributed residuals
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save: Whether to save the plot
    
    Returns:
        Matplotlib figure
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors="white", linewidth=0.5)
    axes[0].axhline(y=0, color="red", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted Rating", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor="white", alpha=0.7, color="#2563eb")
    axes[1].axvline(x=0, color="red", linestyle="--", lw=2)
    axes[1].axvline(x=residuals.mean(), color="green", linestyle="-", lw=2, label=f"Mean: {residuals.mean():.3f}")
    axes[1].set_xlabel("Residuals", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        save_plot(fig, "residual_plot.png")
    
    return fig


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Generate feature importance plot.
    
    Supports:
        - Tree-based models (feature_importances_)
        - Linear models (coef_)
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to show
        save: Whether to save the plot
    
    Returns:
        Matplotlib figure or None
    """
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        logger.warning("Model doesn't support feature importance. Skipping.")
        return None
    
    # Create DataFrame and sort
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    
    # Take top N features
    if len(feat_imp) > top_n:
        feat_imp = feat_imp.tail(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(feat_imp) * 0.4)))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_imp)))
    ax.barh(feat_imp["Feature"], feat_imp["Importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title("Feature Importance", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    
    if save:
        save_plot(fig, "feature_importance.png")
    
    # Log top features
    top_3 = feat_imp.tail(3)["Feature"].tolist()[::-1]
    logger.info(f"Top 3 features: {top_3}")
    
    return fig


# ============================================================================
# EDA VISUALIZATIONS
# ============================================================================

def generate_eda_plots(df: pd.DataFrame) -> None:
    """
    Generate all EDA visualizations for movie dataset.
    
    Plots generated:
        1. Missing values heatmap
        2. Rating distribution
        3. Genre distribution
        4. Movies by year
        5. Duration distribution
        6. Votes distribution
        7. Top directors
        8. Top actors
        9. Correlation heatmap
    
    Args:
        df: DataFrame (raw or partially preprocessed)
    """
    print_section("GENERATING EDA VISUALIZATIONS")
    
    verify_directory(OUTPUTS_DIR)
    sns.set_style("whitegrid")
    
    # 1. Missing Values Heatmap
    _plot_missing_values(df)
    
    # 2. Rating Distribution
    if "Rating" in df.columns:
        _plot_rating_distribution(df)
    
    # 3. Genre Distribution
    if "Genre" in df.columns:
        _plot_genre_distribution(df)
    
    # 4. Movies by Year
    if "Year" in df.columns:
        _plot_movies_by_year(df)
    
    # 5. Duration Distribution
    duration_col = "Duration_Minutes" if "Duration_Minutes" in df.columns else "Duration"
    if duration_col in df.columns:
        _plot_duration_distribution(df, duration_col)
    
    # 6. Votes Distribution
    if "Votes" in df.columns:
        _plot_votes_distribution(df)
    
    # 7. Top Directors
    if "Director" in df.columns:
        _plot_top_directors(df)
    
    # 8. Top Actors
    if "Actor 1" in df.columns:
        _plot_top_actors(df)
    
    # 9. Correlation Heatmap
    _plot_correlation_heatmap(df)
    
    logger.info("All EDA plots generated and saved")


def _plot_missing_values(df: pd.DataFrame) -> None:
    """Plot missing values heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap", fontsize=15, fontweight="bold")
    save_plot(fig, "missing_values_heatmap.png")


def _plot_rating_distribution(df: pd.DataFrame) -> None:
    """Plot rating distribution."""
    ratings = pd.to_numeric(df["Rating"], errors="coerce").dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(ratings, bins=20, kde=True, color="#2563eb", ax=ax)
    ax.axvline(ratings.mean(), color="red", linestyle="--", lw=2, label=f"Mean: {ratings.mean():.2f}")
    ax.axvline(ratings.median(), color="green", linestyle="-", lw=2, label=f"Median: {ratings.median():.2f}")
    ax.set_xlabel("IMDb Rating", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of IMDb Ratings", fontsize=15, fontweight="bold")
    ax.legend()
    save_plot(fig, "rating_distribution.png")


def _plot_genre_distribution(df: pd.DataFrame) -> None:
    """Plot genre distribution (top genres)."""
    # Extract primary genre
    genres = df["Genre"].dropna().apply(lambda x: str(x).split(",")[0].strip())
    genre_counts = genres.value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(genre_counts)))
    bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color=colors)
    ax.set_xlabel("Number of Movies", fontsize=13)
    ax.set_title("Top 15 Genres by Movie Count", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, genre_counts.values[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                str(count), va="center", fontsize=10)
    
    save_plot(fig, "genre_distribution.png")


def _plot_movies_by_year(df: pd.DataFrame) -> None:
    """Plot movies released per year."""
    # Clean year
    years = pd.to_numeric(df["Year"].astype(str).str.extract(r'(\d{4})')[0], errors="coerce")
    years = years[(years >= 1950) & (years <= 2025)].dropna()
    year_counts = years.value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(year_counts.index, year_counts.values, alpha=0.3, color="#2563eb")
    ax.plot(year_counts.index, year_counts.values, color="#2563eb", lw=2)
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Number of Movies", fontsize=13)
    ax.set_title("Movies Released Per Year", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    save_plot(fig, "movies_by_year.png")


def _plot_duration_distribution(df: pd.DataFrame, col: str) -> None:
    """Plot movie duration distribution."""
    if col == "Duration_Minutes":
        durations = df[col].dropna()
    else:
        durations = df[col].astype(str).str.extract(r'(\d+)')[0]
        durations = pd.to_numeric(durations, errors="coerce").dropna()
    
    durations = durations[(durations > 0) & (durations < 400)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(durations, bins=30, kde=True, color="#8b5cf6", ax=ax)
    ax.axvline(durations.mean(), color="red", linestyle="--", lw=2, label=f"Mean: {durations.mean():.0f} min")
    ax.set_xlabel("Duration (minutes)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of Movie Duration", fontsize=15, fontweight="bold")
    ax.legend()
    save_plot(fig, "duration_distribution.png")


def _plot_votes_distribution(df: pd.DataFrame) -> None:
    """Plot votes distribution (log scale)."""
    votes = pd.to_numeric(df["Votes"].astype(str).str.replace(",", ""), errors="coerce").dropna()
    votes = votes[votes > 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.log10(votes + 1), bins=30, edgecolor="white", alpha=0.7, color="#f59e0b")
    ax.set_xlabel("Log10(Votes)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of Votes (Log Scale)", fontsize=15, fontweight="bold")
    save_plot(fig, "votes_distribution.png")


def _plot_top_directors(df: pd.DataFrame) -> None:
    """Plot top directors by movie count."""
    directors = df["Director"].dropna()
    director_counts = directors.value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(director_counts)))
    ax.barh(director_counts.index[::-1], director_counts.values[::-1], color=colors)
    ax.set_xlabel("Number of Movies", fontsize=13)
    ax.set_title("Top 15 Directors by Movie Count", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    save_plot(fig, "top_directors.png")


def _plot_top_actors(df: pd.DataFrame) -> None:
    """Plot top lead actors by appearances."""
    actors = df["Actor 1"].dropna()
    actor_counts = actors.value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(actor_counts)))
    ax.barh(actor_counts.index[::-1], actor_counts.values[::-1], color=colors)
    ax.set_xlabel("Number of Movies", fontsize=13)
    ax.set_title("Top 15 Lead Actors by Movie Count", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    save_plot(fig, "top_actors.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        logger.warning("Not enough numeric columns for correlation heatmap")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    save_plot(fig, "correlation_heatmap.png")
