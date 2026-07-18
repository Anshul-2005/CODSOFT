# 💳 Credit Card Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6%2B-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-4C72B0?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**CodSoft Data Science Internship — Task 5**

*Build a machine learning model to detect fraudulent credit card transactions using classification algorithms on a highly imbalanced dataset.*

[Overview](#-overview) •
[Dataset](#-dataset) •
[Features](#-features) •
[Installation](#-installation) •
[How to Run](#-how-to-run) •
[Results](#-results)

</div>

---

## 📋 Overview

This project builds an end-to-end machine learning pipeline for **Credit Card Fraud Detection**. The system analyzes transaction data to identify potentially fraudulent transactions among legitimate ones. The dataset presents a significant challenge due to its extreme class imbalance — fraudulent transactions represent only **0.172%** of all transactions.

### Key Highlights

- ✅ **Multiple Model Comparison** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- ✅ **Class Imbalance Handling** — Balanced class weights and stratified splitting
- ✅ **Hyperparameter Tuning** — GridSearchCV for optimal model performance
- ✅ **Comprehensive Evaluation** — ROC-AUC, Precision-Recall, Confusion Matrix
- ✅ **Feature Engineering** — Time-based, amount-based, and PCA aggregation features
- ✅ **Production-Ready** — Modular code, model persistence, automated pipeline

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Transactions** | 284,807 |
| **Features** | 31 (V1–V28 PCA, Time, Amount, Class) |
| **Fraud Cases** | 492 (0.172%) |
| **Legitimate Cases** | 284,315 (99.828%) |
| **Imbalance Ratio** | ~578:1 |

### Feature Description

| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed since first transaction |
| `V1–V28` | PCA-transformed features (anonymized) |
| `Amount` | Transaction amount |
| `Class` | Target variable (0 = Legitimate, 1 = Fraud) |

---

## 🔧 Features

### Engineered Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `Hour` | Hour of day from Time | Fraud patterns may vary by time |
| `Is_Night` | Night transaction flag (22:00-06:00) | Unusual timing indicator |
| `Log_Amount` | Log-transformed amount | Reduces skewness |
| `Amount_Bin` | Binned amount categories | Captures amount ranges |
| `V_Mean` | Mean of V1-V28 | Overall PCA pattern |
| `V_Std` | Std deviation of V1-V28 | Pattern variability |
| `V_Extreme_Count` | Count of extreme V values | Anomaly indicator |

---

## 📁 Project Structure

```
Credit_Card_Fraud_Detection/
│
├── dataset/
│   └── download_dataset.md          # Dataset download instructions
│
├── notebook/
│   └── Credit_Card_Fraud_Detection.ipynb  # Jupyter notebook
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── preprocessing.py              # Data preprocessing
│   ├── feature_engineering.py        # Feature creation
│   ├── model_training.py             # Model training & tuning
│   ├── evaluation.py                 # Model evaluation & plots
│   ├── prediction.py                 # Prediction generation
│   └── utils.py                      # Utility functions
│
├── models/
│   ├── fraud_detection_model.pkl     # Trained model
│   └── scaler.pkl                    # Fitted scaler
│
├── outputs/
│   ├── class_distribution.png        # Class distribution plot
│   ├── transaction_amount_distribution.png
│   ├── correlation_heatmap.png       # Feature correlations
│   ├── fraud_vs_legitimate.png       # Class comparison
│   ├── missing_values_heatmap.png    # Missing data analysis
│   ├── feature_importance.png        # Feature importances
│   ├── confusion_matrix.png          # Confusion matrix
│   ├── roc_curve.png                 # ROC curve
│   ├── precision_recall_curve.png    # Precision-Recall curve
│   ├── model_comparison.csv          # Model comparison table
│   └── predictions.csv              # Prediction results
│
├── screenshots/
│   └── README.md                     # Screenshot instructions
│
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── main.py                           # Main pipeline script
├── index.html                        # Landing page
└── .gitignore                        # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the dataset/ folder

# OR use Kaggle CLI:
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d dataset/
```

---

## ▶️ How to Run

### Option 1: Run Complete Pipeline

```bash
python main.py
```

This will automatically:
- ✅ Load and preprocess the dataset
- ✅ Engineer features
- ✅ Train 4 different models
- ✅ Tune the best model with GridSearchCV
- ✅ Generate all visualizations
- ✅ Save model and scaler
- ✅ Generate predictions.csv

### Option 2: Jupyter Notebook

```bash
cd notebook
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

### Option 3: Landing Page

Open `index.html` in your browser to view the project showcase.

---

## 🔄 ML Pipeline Workflow

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  Load Data  │───▶│  Preprocess  │───▶│   Feature     │
│             │    │              │    │  Engineering   │
└─────────────┘    └──────────────┘    └───────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│   Save &    │◀───│   Evaluate   │◀───│    Train      │
│   Export    │    │              │    │   Models       │
└─────────────┘    └──────────────┘    └───────────────┘
```

---

## 🤖 Models Used

| # | Model | Key Characteristics |
|---|-------|-------------------|
| 1 | **Logistic Regression** | Linear model, fast, interpretable, balanced class weights |
| 2 | **Decision Tree** | Non-linear, interpretable, handles imbalance |
| 3 | **Random Forest** | Ensemble of trees, robust, feature importance |
| 4 | **Gradient Boosting** | Sequential boosting, high accuracy, handles complex patterns |

### Class Imbalance Strategy

- **Balanced Class Weights** — Automatically adjusts weights inversely proportional to class frequencies
- **Stratified Splitting** — Maintains fraud ratio in both train and test sets
- **Evaluation with Appropriate Metrics** — Focus on Recall, F1-Score, and ROC-AUC over accuracy

---

## 📈 Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|-------------|---------------|
| **Accuracy** | Overall correct predictions | Can be misleading with imbalanced data |
| **Precision** | Of predicted frauds, how many are real | Minimizes false alarms |
| **Recall** | Of real frauds, how many were caught | Minimizes missed frauds |
| **F1 Score** | Harmonic mean of Precision & Recall | Balances precision and recall |
| **ROC-AUC** | Area under ROC curve | Measures overall discrimination ability |

---

## 📊 Results

### Model Comparison

All models are trained with class balancing and evaluated on the test set. The best model is further tuned using GridSearchCV with 3-fold cross-validation optimizing for F1-Score.

### Generated Visualizations

| Visualization | Description |
|---------------|-------------|
| Class Distribution | Shows extreme imbalance (99.83% vs 0.17%) |
| Transaction Amount | Distribution comparison: fraud vs legitimate |
| Fraud vs Legitimate | Multi-panel feature comparison |
| Correlation Heatmap | Feature correlation analysis |
| Missing Values | Data quality assessment |
| Feature Importance | Top contributing features |
| Confusion Matrix | True/False positive/negative counts |
| ROC Curve | Trade-off between TPR and FPR |
| Precision-Recall Curve | Precision vs recall trade-off |

---

## 🔮 Future Improvements

- [ ] Implement SMOTE / ADASYN for synthetic oversampling
- [ ] Add XGBoost and LightGBM models
- [ ] Implement deep learning approach (Autoencoder for anomaly detection)
- [ ] Real-time fraud detection with streaming data
- [ ] Add model explainability with SHAP values
- [ ] Deploy as REST API with Flask/FastAPI
- [ ] Add cost-sensitive learning (different costs for FP vs FN)
- [ ] Implement isolation forest for unsupervised detection

---

## 🙏 Acknowledgments

- **CodSoft** — For the Data Science Internship opportunity
- **Kaggle** — For hosting the Credit Card Fraud Detection dataset
- **Machine Learning Group (ULB)** — For creating and sharing the dataset
- **Scikit-Learn** — For the comprehensive ML library

---

<div align="center">

**Made with ❤️ for CodSoft Data Science Internship**

*Task 5 — Credit Card Fraud Detection*

</div>
