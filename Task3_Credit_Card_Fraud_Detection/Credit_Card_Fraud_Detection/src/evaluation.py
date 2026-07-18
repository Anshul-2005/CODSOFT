"""
Model Evaluation Module
========================

Handles model evaluation, metrics generation, and visualization
for the Credit Card Fraud Detection project.

Functions:
    - generate_classification_report: Full classification report
    - plot_confusion_matrix: Confusion matrix visualization
    - plot_roc_curve: ROC curve visualization
    - plot_precision_recall_curve: Precision-recall curve
    - plot_feature_importance: Feature importance chart
    - evaluate_model: Complete evaluation pipeline
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

from src.utils import get_project_root, set_plot_style

# Logger
logger = logging.getLogger("CreditCardFraudDetection")


def generate_classification_report(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Generate comprehensive classification report.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        model_name: Name of the model
        
    Returns:
        Dict: Evaluation metrics
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_test, y_prob)
    
    print(f"\n{'='*50}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*50}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Fraud"]
    ))
    
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    logger.info(f"Classification report generated for {model_name}")
    
    return metrics


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Best Model",
    save: bool = True
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save: Whether to save the plot
    """
    set_plot_style()
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
        annot_kws={"size": 14}
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    
    plt.tight_layout()
    
    if save:
        output_path = get_project_root() / "outputs" / "confusion_matrix.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {output_path}")
    
    plt.close()


def plot_roc_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    model_name: str = "Best Model",
    save: bool = True
) -> None:
    """
    Plot and save ROC curve.
    
    Args:
        y_test: True labels
        y_prob: Prediction probabilities
        model_name: Name of the model
        save: Whether to save the plot
    """
    set_plot_style()
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5,
            label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--",
            label="Random Classifier")
    
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_path = get_project_root() / "outputs" / "roc_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {output_path}")
    
    plt.close()


def plot_precision_recall_curve(
    y_test: pd.Series,
    y_prob: np.ndarray,
    model_name: str = "Best Model",
    save: bool = True
) -> None:
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_test: True labels
        y_prob: Prediction probabilities
        model_name: Name of the model
        save: Whether to save the plot
    """
    set_plot_style()
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#7C3AED", lw=2.5,
            label=f"{model_name} (AUC = {pr_auc:.4f})")
    
    ax.fill_between(recall, precision, alpha=0.1, color="#7C3AED")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_path = get_project_root() / "outputs" / "precision_recall_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Precision-Recall curve saved to {output_path}")
    
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "Best Model",
    top_n: int = 20,
    save: bool = True
) -> None:
    """
    Plot and save feature importance chart.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to display
        save: Whether to save the plot
    """
    set_plot_style()
    
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} does not support feature importance")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    bars = ax.barh(
        range(len(top_features)), top_importances[::-1],
        color=colors[::-1], edgecolor="white", linewidth=0.5
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}",
                 fontsize=16, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        output_path = get_project_root() / "outputs" / "feature_importance.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance saved to {output_path}")
    
    plt.close()


def evaluate_model(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str] = None
) -> Dict[str, float]:
    """
    Run complete evaluation pipeline for a model.
    
    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Testing features
        y_test: Testing labels
        feature_names: List of feature names
        
    Returns:
        Dict: Evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    
    # Classification report
    metrics = generate_classification_report(y_test, y_pred, y_prob, model_name)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    # ROC curve
    if y_prob is not None:
        plot_roc_curve(y_test, y_prob, model_name)
        plot_precision_recall_curve(y_test, y_prob, model_name)
    
    # Feature importance
    if feature_names is not None:
        plot_feature_importance(model, feature_names, model_name)
    
    logger.info(f"Evaluation completed for {model_name}")
    
    return metrics
