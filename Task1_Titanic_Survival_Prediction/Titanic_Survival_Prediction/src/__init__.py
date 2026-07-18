"""
Titanic Survival Prediction - Source Package
=============================================

CodSoft Data Science Internship — Task 1

Modules:
    - utils: Utilities, logging, path management
    - preprocessing: Data cleaning, encoding, scaling
    - feature_engineering: Feature creation
    - model_training: Training, comparison, hyperparameter tuning
    - evaluation: Metrics, plots, EDA visualizations
    - prediction: Inference and batch predictions
"""

from src.utils import (
    setup_directories,
    load_dataset,
    dataset_overview,
    setup_logging,
    PROJECT_ROOT,
    DATASET_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
)

from src.preprocessing import (
    full_preprocessing_pipeline,
    encode_categorical,
    drop_unnecessary_columns,
    scale_features,
)

from src.feature_engineering import full_feature_engineering

from src.model_training import (
    split_data,
    train_and_compare,
    hyperparameter_tuning,
    save_model,
    load_model,
)

from src.evaluation import (
    full_evaluation,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    generate_eda_plots,
)

from src.prediction import (
    predict_survival,
    generate_predictions_csv,
    demo_predictions,
)

__version__ = "1.0.0"
__author__ = "CodSoft Data Science Intern"
__task__ = "Task 1 - Titanic Survival Prediction"
