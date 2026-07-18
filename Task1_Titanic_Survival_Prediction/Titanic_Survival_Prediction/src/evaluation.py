"""
Model evaluation module for Titanic Survival Prediction.
Generates metrics, plots, and evaluation reports.

Author: CodSoft Data Science Intern
Task: Task 1 - Titanic Survival Prediction
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.utils import OUTPUTS_DIR, save_plot, print_section, verify_directory

# Get logger
logger = logging.getLogger("TitanicPrediction")


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
    Comprehensive model evaluation with all metrics.
    
    Metrics:
        - Accuracy: Overall correctness
        - Precision: True positives / predicted positives
        - Recall: True positives / actual positives
        - F1 Score: Harmonic mean of precision and recall
        - ROC-AUC: Area under the ROC curve
        - Cross-validation score
    
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
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    
    metrics = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc, 4) if roc_auc else "N/A",
        "CV Mean": round(cv_scores.mean(), 4),
        "CV Std": round(cv_scores.std(), 4),
    }
    
    # Print formatted results
    print(f"\n  Model: {model_name}")
    print(f"  {'─' * 50}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"    → Proportion of correct predictions overall")
    print(f"  Precision:  {prec:.4f}")
    print(f"    → Of predicted survivors, how many actually survived")
    print(f"  Recall:     {rec:.4f}")
    print(f"    → Of actual survivors, how many were identified")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"    → Harmonic mean of precision and recall")
    if roc_auc:
        print(f"  ROC-AUC:    {roc_auc:.4f}")
        print(f"    → Ability to distinguish between classes")
    print(f"  CV Score:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    → Average accuracy across 5 folds")
    
    # Classification Report
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=["Not Survived", "Survived"]
    ))
    
    logger.info(f"Evaluation complete. Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return metrics


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """
    Generate and save confusion matrix heatmap.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save: Whether to save the plot
    
    Returns:
        Matplotlib figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
        ax=ax,
        annot_kws={"size": 16},
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    
    if save:
        save_plot(fig, "confusion_matrix.png")
    
    return fig


# ============================================================================
# ROC CURVE
# ============================================================================

def plot_roc_curve(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Generate and save ROC curve.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        save: Whether to save the plot
    
    Returns:
        Matplotlib figure or None if model doesn't support probabilities
    """
    if not hasattr(model, "predict_proba"):
        logger.warning("Model doesn't support probability estimates. Skipping ROC curve.")
        return None
    
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    if save:
        save_plot(fig, "roc_curve.png")
    
    return fig


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(
    model: Any, 
    feature_names: List[str],
    top_n: int = 20,
    save: bool = True,
) -> Optional[plt.Figure]:
    """
    Generate and save feature importance plot.
    
    Supports:
        - Tree-based models (feature_importances_)
        - Linear models (coef_)
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to show
        save: Whether to save the plot
    
    Returns:
        Matplotlib figure or None if model doesn't support feature importance
    """
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
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
    fig, ax = plt.subplots(figsize=(10, max(6, len(feat_imp) * 0.35)))
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
    Generate all EDA visualizations.
    
    Plots generated:
        1. Missing values heatmap
        2. Correlation heatmap
        3. Survival distribution
        4. Passenger class distribution
        5. Gender distribution
        6. Age distribution
        7. Fare distribution
        8. Survival vs Gender
        9. Survival vs Passenger Class
        10. Age vs Survival
    
    Args:
        df: Raw or preprocessed DataFrame
    """
    print_section("GENERATING EDA VISUALIZATIONS")
    
    verify_directory(OUTPUTS_DIR)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Missing Value Heatmap
    _plot_missing_values(df)
    
    # 2. Correlation Heatmap
    _plot_correlation_heatmap(df)
    
    # 3. Survival Distribution
    if "Survived" in df.columns:
        _plot_survival_distribution(df)
    
    # 4. Passenger Class Distribution
    if "Pclass" in df.columns:
        _plot_pclass_distribution(df)
    
    # 5. Gender Distribution
    if "Sex" in df.columns:
        _plot_gender_distribution(df)
    
    # 6. Age Distribution
    if "Age" in df.columns:
        _plot_age_distribution(df)
    
    # 7. Fare Distribution
    if "Fare" in df.columns:
        _plot_fare_distribution(df)
    
    # 8. Survival vs Gender
    if "Survived" in df.columns and "Sex" in df.columns:
        _plot_survival_vs_gender(df)
    
    # 9. Survival vs Passenger Class
    if "Survived" in df.columns and "Pclass" in df.columns:
        _plot_survival_vs_pclass(df)
    
    # 10. Age vs Survival
    if "Survived" in df.columns and "Age" in df.columns:
        _plot_age_vs_survival(df)
    
    logger.info("All EDA plots generated and saved")


def _plot_missing_values(df: pd.DataFrame) -> None:
    """Plot missing values heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap", fontsize=15, fontweight="bold")
    save_plot(fig, "missing_values_heatmap.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold")
    save_plot(fig, "correlation_heatmap.png")


def _plot_survival_distribution(df: pd.DataFrame) -> None:
    """Plot survival target distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Survived", data=df, palette=["#ef4444", "#22c55e"], ax=ax)
    ax.set_title("Survival Distribution", fontsize=15, fontweight="bold")
    ax.set_xticklabels(["Not Survived (0)", "Survived (1)"])
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=13
        )
    save_plot(fig, "survival_distribution.png")


def _plot_pclass_distribution(df: pd.DataFrame) -> None:
    """Plot passenger class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Pclass", data=df, palette="Set2", ax=ax)
    ax.set_title("Passenger Class Distribution", fontsize=15, fontweight="bold")
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=13
        )
    save_plot(fig, "pclass_distribution.png")


def _plot_gender_distribution(df: pd.DataFrame) -> None:
    """Plot gender distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Sex", data=df, palette=["#3b82f6", "#ec4899"], ax=ax)
    ax.set_title("Gender Distribution", fontsize=15, fontweight="bold")
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=13
        )
    save_plot(fig, "gender_distribution.png")


def _plot_age_distribution(df: pd.DataFrame) -> None:
    """Plot age distribution with KDE."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Age"].dropna(), bins=30, kde=True, color="#8b5cf6", ax=ax)
    ax.set_title("Age Distribution", fontsize=15, fontweight="bold")
    median_age = df["Age"].median()
    ax.axvline(median_age, color="red", linestyle="--", label=f'Median: {median_age:.1f}')
    ax.legend()
    save_plot(fig, "age_distribution.png")


def _plot_fare_distribution(df: pd.DataFrame) -> None:
    """Plot fare distribution with KDE."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Fare"].dropna(), bins=30, kde=True, color="#f59e0b", ax=ax)
    ax.set_title("Fare Distribution", fontsize=15, fontweight="bold")
    median_fare = df["Fare"].median()
    ax.axvline(median_fare, color="red", linestyle="--", label=f'Median: {median_fare:.1f}')
    ax.legend()
    save_plot(fig, "fare_distribution.png")


def _plot_survival_vs_gender(df: pd.DataFrame) -> None:
    """Plot survival rate by gender."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Sex", hue="Survived", data=df, palette=["#ef4444", "#22c55e"], ax=ax)
    ax.set_title("Survival vs Gender", fontsize=15, fontweight="bold")
    ax.legend(["Not Survived", "Survived"])
    save_plot(fig, "survival_vs_gender.png")


def _plot_survival_vs_pclass(df: pd.DataFrame) -> None:
    """Plot survival rate by passenger class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Pclass", hue="Survived", data=df, palette=["#ef4444", "#22c55e"], ax=ax)
    ax.set_title("Survival vs Passenger Class", fontsize=15, fontweight="bold")
    ax.legend(["Not Survived", "Survived"])
    save_plot(fig, "survival_vs_pclass.png")


def _plot_age_vs_survival(df: pd.DataFrame) -> None:
    """Plot age distribution by survival status."""
    fig, ax = plt.subplots(figsize=(10, 5))
    survived = df[df["Survived"] == 1]["Age"].dropna()
    not_survived = df[df["Survived"] == 0]["Age"].dropna()
    ax.hist(not_survived, bins=30, alpha=0.6, color="#ef4444", label="Not Survived", edgecolor="white")
    ax.hist(survived, bins=30, alpha=0.6, color="#22c55e", label="Survived", edgecolor="white")
    ax.set_xlabel("Age", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Age vs Survival", fontsize=15, fontweight="bold")
    ax.legend()
    save_plot(fig, "age_vs_survival.png")
