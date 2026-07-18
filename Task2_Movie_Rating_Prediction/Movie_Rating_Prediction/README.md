# 🎬 Movie Rating Prediction

**CodSoft Data Science Internship — Task 2**

A complete Machine Learning pipeline that predicts IMDb movie ratings using regression algorithms.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Information](#-dataset-information)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [ML Pipeline](#-ml-pipeline)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Output Files](#-output-files)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)

---

## 📋 Project Overview

This project builds a regression model to predict IMDb movie ratings based on various movie attributes. The model analyzes features like genre, director, actors, duration, votes, and release year to estimate how well a movie will be rated.

### Key Features

- ✅ Complete end-to-end ML regression pipeline
- ✅ 6 regression algorithms compared
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Comprehensive feature engineering
- ✅ 12+ EDA visualizations
- ✅ Detailed evaluation metrics (MAE, RMSE, R²)
- ✅ Modular, reusable code architecture

---

## 🎯 Problem Statement

**Goal:** Build a regression model that predicts the IMDb rating of an Indian movie based on its attributes.

- **Input:** Movie features (genre, director, actors, duration, votes, year)
- **Output:** Predicted IMDb rating (continuous value 0-10)
- **Type:** Regression problem
- **Metrics:** MAE, MSE, RMSE, R² Score

### Business Value

- Help production houses estimate potential movie success
- Assist streaming platforms in content curation
- Enable data-driven decisions in film industry
- Identify factors that contribute to higher ratings

---

## 📊 Dataset Information

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — IMDb India Movies](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) |
| **Rows** | ~15,000+ movies |
| **Columns** | 10 |
| **Target** | `Rating` (0-10 scale) |
| **Format** | CSV |
| **License** | CC0: Public Domain |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| Name | string | Movie title |
| Year | string | Release year |
| Duration | string | Runtime (e.g., "120 min") |
| Genre | string | Movie genre(s), comma-separated |
| Rating | float | **TARGET** — IMDb rating |
| Votes | string | Number of user votes |
| Director | string | Director name |
| Actor 1 | string | Lead actor |
| Actor 2 | string | Supporting actor |
| Actor 3 | string | Supporting actor |

---

## 📁 Folder Structure

```
Movie_Rating_Prediction/
│
├── dataset/
│   ├── download_dataset.md         # Dataset download instructions
│   └── IMDb_Movies_India.csv       # ← Place dataset here
│
├── notebook/
│   └── Movie_Rating_Prediction.ipynb   # Jupyter notebook (full analysis)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Utilities, logging, path management
│   ├── preprocessing.py            # Data cleaning, encoding, scaling
│   ├── feature_engineering.py      # Feature creation and transformation
│   ├── model_training.py           # Training, comparison, tuning
│   ├── evaluation.py               # Metrics, plots, EDA
│   └── prediction.py               # Inference module
│
├── models/
│   ├── movie_rating_model.pkl      # Saved best model
│   └── scaler.pkl                  # Saved StandardScaler
│
├── outputs/
│   ├── rating_distribution.png
│   ├── genre_distribution.png
│   ├── movies_by_year.png
│   ├── duration_distribution.png
│   ├── votes_distribution.png
│   ├── top_directors.png
│   ├── top_actors.png
│   ├── correlation_heatmap.png
│   ├── missing_values_heatmap.png
│   ├── feature_importance.png
│   ├── prediction_vs_actual.png
│   ├── residual_plot.png
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
git clone https://github.com/yourusername/Movie_Rating_Prediction.git
cd Movie_Rating_Prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Go to: https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies
# Download and place CSV file in the 'dataset/' folder
```

---

## 🚀 How to Run

### Run the Complete Pipeline

```bash
python main.py
```

This single command will:
1. Load and explore the dataset
2. Generate 12+ EDA visualizations
3. Clean and preprocess data
4. Engineer predictive features
5. Train 6 regression models
6. Compare models with cross-validation
7. Tune hyperparameters on top models
8. Evaluate the best model
9. Save model and outputs
10. Run sample predictions

### Run the Jupyter Notebook

```bash
cd notebook
jupyter notebook Movie_Rating_Prediction.ipynb
```

---

## 🔬 ML Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Load Data  │ →  │     EDA     │ →  │    Clean    │ →  │  Engineer   │
│   (CSV)     │    │   & Plots   │    │  (Preproc)  │    │  Features   │
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
| Year | Extract first 4-digit year | Handle ranges like "2019-2020" |
| Duration | Extract numeric minutes | Parse "120 min" format |
| Votes | Remove commas, convert to int | Handle "1,234,567" format |
| Rating | Filter valid 0-10 range | Remove outliers/errors |
| Missing | Drop for target, impute for features | Preserve data quality |

### Feature Engineering

| Feature | Description | Rationale |
|---------|-------------|-----------|
| **PrimaryGenre** | First listed genre | Simplify multi-genre movies |
| **GenreCount** | Number of genres | Multi-genre patterns |
| **Director_Encoded** | Top 50 directors encoded | Director reputation |
| **Director_AvgRating** | Mean rating per director | Director quality signal |
| **LeadActor_Encoded** | Top 100 actors encoded | Star power |
| **LeadActor_AvgRating** | Mean rating per actor | Actor quality signal |
| **Decade** | Year // 10 * 10 | Era patterns |
| **MovieAge** | Current year - release year | Age effect on ratings |
| **VoteCategory** | Binned vote counts | Popularity signal |
| **Votes_Log** | Log-transformed votes | Normalize skewed distribution |
| **DurationCategory** | Short/Standard/Long/VeryLong | Runtime patterns |

---

## 🤖 Models Used

| # | Model | Description |
|---|-------|-------------|
| 1 | **Linear Regression** | Simple linear baseline |
| 2 | **Ridge Regression** | L2 regularized linear model |
| 3 | **Lasso Regression** | L1 regularized linear model |
| 4 | **Decision Tree** | Non-linear, interpretable |
| 5 | **Random Forest** | Ensemble, reduces overfitting |
| 6 | **Gradient Boosting** | Sequential ensemble, often best |

---

## 📏 Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average rating error |
| **MSE** | Mean Squared Error | Penalizes large errors |
| **RMSE** | Root Mean Squared Error | Same units as rating |
| **R²** | Coefficient of Determination | Variance explained (0-1) |
| **CV R²** | Cross-Validation R² | Generalization measure |

### Interpretation Guide

- **RMSE < 1.0**: Excellent — predictions within 1 rating point
- **RMSE 1.0-1.5**: Good — reasonable accuracy
- **RMSE > 1.5**: Needs improvement
- **R² > 0.5**: Good explanatory power
- **R² > 0.7**: Excellent model fit

---

## 📈 Results

The best model is automatically selected based on cross-validation R² score.

> **Note:** Actual metrics are generated after running `python main.py`.

### Expected Performance Range

| Metric | Typical Range |
|--------|---------------|
| MAE | 0.6-0.9 |
| RMSE | 0.8-1.2 |
| R² | 0.35-0.55 |

### Key Findings

- **Votes** is the strongest predictor (more votes → often higher ratings)
- **Director reputation** significantly impacts ratings
- **Genre** influences ratings (some genres consistently rate higher)
- **Duration** has a modest positive correlation with ratings
- **Year/Age** shows interesting patterns (classics vs. recent films)

---

## 📄 Output Files

After running `python main.py`, these files are generated:

### outputs/

| File | Description |
|------|-------------|
| `model_comparison.csv` | Performance comparison of all 6 models |
| `predictions.csv` | Test set predictions with residuals |
| `rating_distribution.png` | Distribution of IMDb ratings |
| `genre_distribution.png` | Top genres by movie count |
| `movies_by_year.png` | Movies released per year |
| `duration_distribution.png` | Movie duration distribution |
| `votes_distribution.png` | Vote count distribution (log scale) |
| `top_directors.png` | Most prolific directors |
| `top_actors.png` | Most appearing lead actors |
| `correlation_heatmap.png` | Feature correlations |
| `missing_values_heatmap.png` | Missing data visualization |
| `feature_importance.png` | Feature importance from best model |
| `prediction_vs_actual.png` | Scatter plot of predictions |
| `residual_plot.png` | Residual analysis |

### models/

| File | Description |
|------|-------------|
| `movie_rating_model.pkl` | Serialized best model |
| `scaler.pkl` | Fitted StandardScaler |

---

## 🔮 Future Improvements

1. **More Features:** Extract budget, box office, release month
2. **NLP Features:** Sentiment analysis of movie descriptions
3. **Advanced Models:** XGBoost, LightGBM, Neural Networks
4. **Ensemble Methods:** Stacking, blending multiple models
5. **Target Engineering:** Predict rating categories instead
6. **Time Series:** Analyze rating trends over time
7. **External Data:** Integrate social media sentiment
8. **Feature Selection:** Recursive feature elimination
9. **Hyperparameter Optimization:** Bayesian optimization
10. **Deployment:** Create API for real-time predictions

---

## 📸 Screenshots

Screenshots of key outputs can be found in the `screenshots/` folder after running the pipeline.

Expected screenshots:
- Rating distribution
- Model comparison output
- Feature importance plot
- Prediction vs actual scatter
- Terminal output

---

## 🙏 Acknowledgements

- **Dataset:** [Adrian McMahon on Kaggle](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)
- **Program:** CodSoft Data Science Internship
- **Tools:** Python, Pandas, Scikit-Learn, Matplotlib, Seaborn

---

## 📄 License

This project is developed as part of the **CodSoft Data Science Internship** program.

Free for educational and portfolio use.

---

## 👤 Author

**CodSoft Data Science Intern**

- Task: Task 2 — Movie Rating Prediction
- Program: CodSoft Data Science Internship

---

*Built with Python, Pandas, NumPy, Scikit-Learn, Matplotlib, and Seaborn*
