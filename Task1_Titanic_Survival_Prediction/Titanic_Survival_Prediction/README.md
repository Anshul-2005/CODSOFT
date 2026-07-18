# 🚢 Titanic Survival Prediction

**CodSoft Data Science Internship — Task 1**

A complete Machine Learning pipeline that predicts passenger survival on the RMS Titanic using classification algorithms.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [ML Pipeline](#-ml-pipeline)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Output Files](#-output-files)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 📋 Project Overview

The sinking of the RMS Titanic on April 15, 1912, is one of the most infamous shipwrecks in history. Of the 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest maritime disasters in history.

This project builds a predictive model to answer: **"What sorts of people were more likely to survive?"** using passenger data such as name, age, gender, socio-economic class, and more.

### Key Features

- ✅ Complete end-to-end ML pipeline
- ✅ 6 classification algorithms compared
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ 7 engineered features
- ✅ 13+ EDA visualizations
- ✅ Comprehensive evaluation metrics
- ✅ Modular, reusable code architecture

---

## 🎯 Business Problem

**Goal:** Build a binary classification model that predicts whether a given passenger survived the Titanic disaster.

- **Input:** Passenger features (class, gender, age, fare, family details, embarkation port)
- **Output:** Survival prediction (0 = Not Survived, 1 = Survived)
- **Metric:** Accuracy, Precision, Recall, F1 Score, ROC-AUC

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Titanic Dataset (yasserh)](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| **Rows** | 891 |
| **Columns** | 12 |
| **Target** | `Survived` (0 = No, 1 = Yes) |
| **Format** | CSV |

### Features

| Column | Type | Description | Missing |
|--------|------|-------------|---------|
| PassengerId | int | Unique identifier | 0% |
| Survived | int | Survival (0 = No, 1 = Yes) — **TARGET** | 0% |
| Pclass | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | 0% |
| Name | string | Passenger name | 0% |
| Sex | string | Gender | 0% |
| Age | float | Age in years | 19.9% |
| SibSp | int | # of siblings/spouses aboard | 0% |
| Parch | int | # of parents/children aboard | 0% |
| Ticket | string | Ticket number | 0% |
| Fare | float | Passenger fare | 0% |
| Cabin | string | Cabin number | 77.1% |
| Embarked | string | Port of embarkation (C/Q/S) | 0.2% |

---

## 📁 Folder Structure

```
Titanic_Survival_Prediction/
│
├── dataset/
│   ├── download_dataset.md         # Dataset download instructions
│   └── Titanic-Dataset.csv         # ← Place dataset here
│
├── notebook/
│   └── Titanic_Survival_Prediction.ipynb   # Jupyter notebook (full analysis)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Utilities, logging, path management
│   ├── preprocessing.py            # Data cleaning, encoding, scaling
│   ├── feature_engineering.py      # Feature creation
│   ├── model_training.py           # Training, comparison, tuning
│   ├── evaluation.py               # Metrics, plots, EDA
│   └── prediction.py               # Inference module
│
├── models/
│   ├── titanic_model.pkl           # Saved best model
│   └── scaler.pkl                  # Saved StandardScaler
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── correlation_heatmap.png
│   ├── missing_values_heatmap.png
│   ├── survival_distribution.png
│   ├── pclass_distribution.png
│   ├── gender_distribution.png
│   ├── age_distribution.png
│   ├── fare_distribution.png
│   ├── survival_vs_gender.png
│   ├── survival_vs_pclass.png
│   ├── age_vs_survival.png
│   ├── predictions.csv
│   └── model_comparison.csv
│
├── screenshots/
│   └── README.md                   # Screenshot documentation
│
├── main.py                         # Complete pipeline entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Titanic_Survival_Prediction.git
cd Titanic_Survival_Prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Go to: https://www.kaggle.com/datasets/yasserh/titanic-dataset
# Download and place 'Titanic-Dataset.csv' in the 'dataset/' folder
```

---

## 🚀 Usage

### Run the Complete Pipeline

```bash
python main.py
```

This single command will:
1. Load and explore the dataset
2. Generate 13+ EDA visualizations
3. Clean and preprocess data
4. Engineer 7 new features
5. Train 6 ML models
6. Compare models with cross-validation
7. Tune hyperparameters on top models
8. Evaluate the best model
9. Save model and outputs
10. Run sample predictions

### Run the Jupyter Notebook

```bash
cd notebook
jupyter notebook Titanic_Survival_Prediction.ipynb
```

---

## 🔬 ML Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Load Data  │ →  │     EDA     │ →  │    Clean    │ →  │  Engineer   │
│   (CSV)     │    │   & Plots   │    │  (Impute)   │    │  Features   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│    Save     │ ←  │  Evaluate   │ ←  │    Tune     │ ←  │   Train     │
│   Model     │    │  (Metrics)  │    │ (GridSearch)│    │  6 Models   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Data Preprocessing

| Step | Strategy | Rationale |
|------|----------|-----------|
| Missing Age | Median by Pclass & Sex | Age varies by class and gender |
| Missing Embarked | Mode (Southampton) | Only 2 missing, mode is reliable |
| Missing Cabin | Binary flag (CabinAvailable) | 77% missing, can't impute |
| Encoding (Sex) | Label Encoding | Binary variable |
| Encoding (Embarked) | One-Hot Encoding | 3 categories, no ordinal relationship |
| Scaling | StandardScaler | Required for KNN, SVM, Logistic Regression |

### Engineered Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| **Title** | Extracted from Name | Encodes social status, gender, age |
| **FamilySize** | SibSp + Parch + 1 | Family size affects survival dynamics |
| **IsAlone** | 1 if FamilySize == 1 | Solo travelers had lower survival |
| **CabinAvailable** | 1 if Cabin recorded | Proxy for socio-economic status |
| **TicketFrequency** | Count of shared tickets | Group travel indicator |
| **FareCategory** | Quartile bins | Reduces outlier noise |
| **AgeCategory** | Child/Teen/Adult/Senior | Captures non-linear age effects |

---

## 🤖 Models Used

| # | Model | Description |
|---|-------|-------------|
| 1 | **Logistic Regression** | Linear baseline, interpretable |
| 2 | **Decision Tree** | Non-linear, prone to overfitting |
| 3 | **Random Forest** | Ensemble, reduces overfitting via bagging |
| 4 | **Gradient Boosting** | Sequential ensemble, often highest accuracy |
| 5 | **K-Nearest Neighbors** | Instance-based, sensitive to scaling |
| 6 | **Support Vector Machine** | Margin-based, effective in high dimensions |

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall proportion of correct predictions |
| **Precision** | Of predicted survivors, how many actually survived |
| **Recall** | Of actual survivors, how many were correctly identified |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Model's ability to distinguish between classes |
| **Cross-Validation** | 5-fold CV for generalization assessment |

---

## 📈 Results

The best model is automatically selected based on cross-validation performance and further refined via GridSearchCV hyperparameter tuning.

> **Note:** Actual metrics are generated after running `python main.py`. Results depend on random state and dataset specifics.

### Expected Performance Range

| Metric | Typical Range |
|--------|---------------|
| Accuracy | 80-85% |
| Precision | 75-82% |
| Recall | 70-80% |
| F1 Score | 75-82% |
| ROC-AUC | 0.85-0.90 |

### Key Findings

- **Gender** is the strongest predictor — females survived at much higher rates
- **Passenger Class** significantly affects survival — 1st class had highest survival
- **Age** matters — children had priority and better survival chances
- **Fare** correlates positively with survival (proxy for wealth/cabin location)
- **Family Size** shows a sweet spot — small families survived better

---

## 📄 Output Files

After running `python main.py`, the following files are generated:

### outputs/

| File | Description |
|------|-------------|
| `model_comparison.csv` | Performance comparison of all 6 models |
| `predictions.csv` | Test set predictions with probabilities |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC score |
| `feature_importance.png` | Feature importance bar chart |
| `correlation_heatmap.png` | Feature correlations |
| `missing_values_heatmap.png` | Missing data visualization |
| `survival_distribution.png` | Target variable distribution |
| `pclass_distribution.png` | Passenger class distribution |
| `gender_distribution.png` | Gender distribution |
| `age_distribution.png` | Age histogram with KDE |
| `fare_distribution.png` | Fare histogram with KDE |
| `survival_vs_gender.png` | Survival rate by gender |
| `survival_vs_pclass.png` | Survival rate by class |
| `age_vs_survival.png` | Age distribution by survival |

### models/

| File | Description |
|------|-------------|
| `titanic_model.pkl` | Serialized best model (Joblib) |
| `scaler.pkl` | Fitted StandardScaler |

---

## 🔮 Future Improvements

1. **Advanced Ensembles:** XGBoost, LightGBM, CatBoost
2. **Stacking/Blending:** Meta-learner combining multiple models
3. **SHAP Analysis:** Explainable AI for individual predictions
4. **More Feature Engineering:** Deck extraction, surname-based family grouping
5. **Neural Networks:** Simple feedforward network comparison
6. **Feature Selection:** RFE, Boruta for optimal subset
7. **Class Imbalance:** SMOTE, class weights if needed
8. **Cross-Validation:** Repeated stratified K-Fold
9. **Deployment:** Streamlit app for interactive predictions

---

## 📄 License

This project is developed as part of the **CodSoft Data Science Internship** program.

Free for educational and portfolio use.

---

## 👤 Author

**CodSoft Data Science Intern**

- Task: Task 1 — Titanic Survival Prediction
- Program: CodSoft Data Science Internship

---

*Built with Python, Pandas, NumPy, Scikit-Learn, Matplotlib, and Seaborn*
