"""
Credit Card Fraud Detection — Main Pipeline
=============================================

CodSoft Data Science Internship — Task 5

This script runs the complete ML pipeline:
    1. Load dataset
    2. Preprocess data
    3. Engineer features
    4. Train multiple models
    5. Tune best model
    6. Evaluate final model
    7. Generate all visualizations
    8. Generate CSV outputs
    9. Save trained model

Usage:
    python main.py

Author: CodSoft Intern
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    setup_logging,
    create_directories,
    get_project_root,
    set_plot_style,
    print_section_header,
)
from src.preprocessing import (
    load_dataset,
    dataset_overview,
    check_missing_values,
    check_duplicates,
    analyze_class_distribution,
    verify_data_types,
    split_data,
    scale_features,
)
from src.feature_engineering import engineer_features
from src.model_training import (
    train_all_models,
    tune_best_model,
    save_model,
)
from src.evaluation import evaluate_model
from src.prediction import (
    predict,
    generate_predictions_csv,
)


def generate_eda_visualizations(df: pd.DataFrame) -> None:
    """
    Generate all EDA visualizations and save to outputs/.
    
    Args:
        df: Cleaned DataFrame
    """
    set_plot_style()
    output_dir = get_project_root() / "outputs"
    logger = logging.getLogger("CreditCardFraudDetection")
    
    # ── 1. Class Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_counts = df["Class"].value_counts()
    colors = ["#2563EB", "#EF4444"]
    labels = ["Legitimate", "Fraud"]
    
    axes[0].bar(labels, class_counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Class Distribution (Count)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=12)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    
    axes[1].pie(
        class_counts.values, labels=labels, colors=colors,
        autopct="%1.3f%%", startangle=90,
        textprops={"fontsize": 12}, explode=(0, 0.1)
    )
    axes[1].set_title("Class Distribution (%)", fontsize=14, fontweight="bold")
    
    plt.suptitle("Transaction Class Distribution", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: class_distribution.png")
    
    # ── 2. Transaction Amount Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df[df["Class"] == 0]["Amount"], bins=80, color="#2563EB",
                 alpha=0.7, label="Legitimate", edgecolor="white")
    axes[0].hist(df[df["Class"] == 1]["Amount"], bins=80, color="#EF4444",
                 alpha=0.7, label="Fraud", edgecolor="white")
    axes[0].set_title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Amount ($)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(0, 500)
    
    axes[1].hist(np.log1p(df[df["Class"] == 0]["Amount"]), bins=60, color="#2563EB",
                 alpha=0.7, label="Legitimate", edgecolor="white")
    axes[1].hist(np.log1p(df[df["Class"] == 1]["Amount"]), bins=60, color="#EF4444",
                 alpha=0.7, label="Fraud", edgecolor="white")
    axes[1].set_title("Log Transaction Amount Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Log(Amount + 1)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "transaction_amount_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: transaction_amount_distribution.png")
    
    # ── 3. Fraud vs Legitimate Transactions ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Amount comparison
    legitimate = df[df["Class"] == 0]["Amount"]
    fraud = df[df["Class"] == 1]["Amount"]
    
    axes[0, 0].boxplot([legitimate, fraud], labels=["Legitimate", "Fraud"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#2563EB", alpha=0.6),
                        medianprops=dict(color="red", linewidth=2))
    axes[0, 0].set_title("Amount: Legitimate vs Fraud", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("Amount ($)", fontsize=11)
    axes[0, 0].set_ylim(0, 500)
    
    # Time distribution
    axes[0, 1].hist(df[df["Class"] == 0]["Time"], bins=48, color="#2563EB",
                     alpha=0.6, label="Legitimate", density=True, edgecolor="white")
    axes[0, 1].hist(df[df["Class"] == 1]["Time"], bins=48, color="#EF4444",
                     alpha=0.6, label="Fraud", density=True, edgecolor="white")
    axes[0, 1].set_title("Time Distribution by Class", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Time (seconds)", fontsize=11)
    axes[0, 1].set_ylabel("Density", fontsize=11)
    axes[0, 1].legend(fontsize=10)
    
    # Key V features (V14, V17 are often important for fraud)
    for idx, v_col in enumerate(["V14", "V17"]):
        ax = axes[1, idx]
        ax.hist(df[df["Class"] == 0][v_col], bins=60, color="#2563EB",
                alpha=0.6, label="Legitimate", density=True, edgecolor="white")
        ax.hist(df[df["Class"] == 1][v_col], bins=60, color="#EF4444",
                alpha=0.6, label="Fraud", density=True, edgecolor="white")
        ax.set_title(f"{v_col} Distribution by Class", fontsize=13, fontweight="bold")
        ax.set_xlabel(v_col, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=10)
    
    plt.suptitle("Fraud vs Legitimate Transactions", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "fraud_vs_legitimate.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: fraud_vs_legitimate.png")
    
    # ── 4. Correlation Heatmap ──
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Select subset of features for readable heatmap
    selected_cols = ["V1", "V2", "V3", "V4", "V5", "V10", "V11", "V12",
                     "V14", "V16", "V17", "V18", "Time", "Amount", "Class"]
    corr_matrix = df[selected_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Correlation Heatmap (Selected Features)", fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: correlation_heatmap.png")
    
    # ── 5. Missing Values Heatmap ──
    fig, ax = plt.subplots(figsize=(14, 6))
    
    missing_data = df.isnull().sum().values.reshape(1, -1)
    
    sns.heatmap(
        missing_data,
        yticklabels=["Missing Count"],
        xticklabels=df.columns,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 7}
    )
    ax.set_title("Missing Values Heatmap", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: missing_values_heatmap.png")


def main():
    """
    Main pipeline execution function.
    """
    # ── Setup ──
    logger = setup_logging()
    create_directories()
    
    print_section_header("CREDIT CARD FRAUD DETECTION")
    print("CodSoft Data Science Internship — Task 5")
    print("=" * 60)
    
    # ── Step 1: Load Dataset ──
    print_section_header("STEP 1: Loading Dataset")
    df = load_dataset()
    
    # ── Step 2: Dataset Overview ──
    print_section_header("STEP 2: Dataset Overview")
    overview = dataset_overview(df)
    
    # ── Step 3: Missing Values ──
    print_section_header("STEP 3: Missing Value Analysis")
    missing_df = check_missing_values(df)
    
    # ── Step 4: Duplicate Detection ──
    print_section_header("STEP 4: Duplicate Detection")
    df = check_duplicates(df, remove=True)
    
    # ── Step 5: Class Distribution ──
    print_section_header("STEP 5: Class Distribution Analysis")
    distribution = analyze_class_distribution(df)
    
    # ── Step 6: Data Type Verification ──
    print_section_header("STEP 6: Data Type Verification")
    dtype_summary = verify_data_types(df)
    print(dtype_summary.to_string(index=False))
    
    # ── Step 7: Generate EDA Visualizations ──
    print_section_header("STEP 7: Generating EDA Visualizations")
    generate_eda_visualizations(df)
    
    # ── Step 8: Train/Test Split ──
    print_section_header("STEP 8: Train/Test Split")
    X_train, X_test, y_train, y_test = split_data(df)
    
    # ── Step 9: Feature Scaling ──
    print_section_header("STEP 9: Feature Scaling")
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    # ── Step 10: Feature Engineering ──
    print_section_header("STEP 10: Feature Engineering")
    X_train, X_test, new_features = engineer_features(X_train, X_test)
    
    # ── Step 11: Train All Models ──
    print_section_header("STEP 11: Training Models")
    results, comparison_df = train_all_models(X_train, y_train, X_test, y_test)
    
    # ── Step 12: Select Best Model ──
    print_section_header("STEP 12: Selecting Best Model")
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info(f"Best model (by F1): {best_model_name}")
    print(f"Best Model: {best_model_name}")
    
    # ── Step 13: Hyperparameter Tuning ──
    print_section_header("STEP 13: Hyperparameter Tuning")
    tuned_model, best_params = tune_best_model(
        best_model_name, X_train, y_train, X_test, y_test
    )
    
    # ── Step 14: Evaluate Tuned Model ──
    print_section_header("STEP 14: Evaluating Tuned Model")
    feature_names = list(X_test.columns)
    metrics = evaluate_model(
        tuned_model, f"Tuned {best_model_name}",
        X_test, y_test, feature_names
    )
    
    # ── Step 15: Generate Predictions ──
    print_section_header("STEP 15: Generating Predictions")
    y_pred, y_prob = predict(tuned_model, X_test)
    predictions_df = generate_predictions_csv(y_test, y_pred, y_prob)
    
    # ── Step 16: Save Model ──
    print_section_header("STEP 16: Saving Model")
    save_model(tuned_model, "fraud_detection_model.pkl")
    
    # ── Summary ──
    print_section_header("PIPELINE COMPLETE")
    print(f"\n  Best Model: {best_model_name}")
    print(f"  Best Parameters: {best_params}")
    print(f"\n  Final Metrics:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.4f}")
    
    print(f"\n  Generated Outputs:")
    output_dir = get_project_root() / "outputs"
    for f in sorted(output_dir.iterdir()):
        if f.name != ".gitkeep":
            print(f"    ✓ {f.name}")
    
    print(f"\n  Saved Models:")
    model_dir = get_project_root() / "models"
    for f in sorted(model_dir.iterdir()):
        if f.name != ".gitkeep":
            print(f"    ✓ {f.name}")
    
    print("\n" + "=" * 60)
    print("  Credit Card Fraud Detection — Pipeline Complete!")
    print("  CodSoft Data Science Internship — Task 5")
    print("=" * 60)


if __name__ == "__main__":
    main()
