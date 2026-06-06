# =========================================================
# FILE: evaluate_model.py
# =========================================================

# Objective: Load trained fraud detection models, execute operational evaluation workflows, compare model performance, analyze threshold behavior,
# and persist evaluation artifacts for production-oriented decision making.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (

    average_precision_score,
    roc_auc_score,

    precision_score,
    recall_score,
    f1_score,

    confusion_matrix,
    classification_report
)

# =========================================================
# DEFINE LOCAL PATHS
# =========================================================

MODEL_DIR = "models"

EVALUATION_DIR = "evaluation"

TRAIN_TEST_SPLIT_PATH = (
    "models/train_test_split.pkl"
)

LOGISTIC_MODEL_PATH = (
    "models/logistic_regression.pkl"
)

RANDOM_FOREST_MODEL_PATH = (
    "models/random_forest.pkl"
)

XGBOOST_MODEL_PATH = (
    "models/xgboost.pkl"
)

LIGHTGBM_MODEL_PATH = (
    "models/lightgbm.pkl"
)

METRICS_OUTPUT_PATH = (
    "evaluation/model_metrics.csv"
)

THRESHOLD_OUTPUT_PATH = (
    "evaluation/threshold_analysis.csv"
)

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():

    """
    Create evaluation directory.
    """

    os.makedirs(EVALUATION_DIR, exist_ok=True)

    print("[INFO] Evaluation directory verified.")

# =========================================================
# LOAD TRAIN/TEST SPLIT
# =========================================================

def load_train_test_split():

    """
    Load persisted train/test split artifacts.
    """

    print("[INFO] Loading train/test split...")

    split_artifacts = joblib.load(
        TRAIN_TEST_SPLIT_PATH
    )

    print("[INFO] Train/test split loaded successfully.")

    return (

        split_artifacts["X_train"],
        split_artifacts["X_test"],

        split_artifacts["y_train"],
        split_artifacts["y_test"]
    )

# =========================================================
# LOAD TRAINED MODELS
# =========================================================

def load_models():

    """
    Load persisted trained models.
    """

    print("[INFO] Loading trained models...")

    models = {

        "Logistic Regression": joblib.load(
            LOGISTIC_MODEL_PATH
        ),

        "Random Forest": joblib.load(
            RANDOM_FOREST_MODEL_PATH
        ),

        "XGBoost": joblib.load(
            XGBOOST_MODEL_PATH
        ),

        "LightGBM": joblib.load(
            LIGHTGBM_MODEL_PATH
        )
    }

    print("[INFO] Models loaded successfully.")

    return models

# =========================================================
# COMPUTE CORE METRICS
# =========================================================

def compute_core_metrics(

    model,
    model_name,

    X_test,
    y_test
):

    """
    Compute operational fraud metrics.
    """

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {

        "Model": model_name,

        "PR_AUC": average_precision_score(
            y_test,
            probabilities
        ),

        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "Precision": precision_score(
            y_test,
            predictions
        ),

        "Recall": recall_score(
            y_test,
            predictions
        ),

        "F1_Score": f1_score(
            y_test,
            predictions
        )
    }

    return metrics

# =========================================================
# DISPLAY CONFUSION MATRIX
# =========================================================

def display_confusion_matrix(

    model,
    model_name,

    X_test,
    y_test
):

    """
    Display confusion matrix.
    """

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n========================================")
    print(f"{model_name.upper()} CONFUSION MATRIX")
    print("========================================")

    print(matrix)

# =========================================================
# DISPLAY CLASSIFICATION REPORT
# =========================================================

def display_classification_report(

    model,
    model_name,

    X_test,
    y_test
):

    """
    Display classification report.
    """

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    report = classification_report(

        y_test,
        predictions
    )

    print("\n========================================")
    print(f"{model_name.upper()} CLASSIFICATION REPORT")
    print("========================================")

    print(report)

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

def threshold_analysis(

    model,
    model_name,

    X_test,
    y_test
):

    """
    Analyze operational threshold behavior.
    """

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    threshold_results = []

    thresholds = np.arange(

        0.10,
        0.91,
        0.10
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        threshold_results.append({

            "Model": model_name,

            "Threshold": threshold,

            "Precision": precision,

            "Recall": recall,

            "F1_Score": f1
        })

    return pd.DataFrame(
        threshold_results
    )

# =========================================================
# PERSIST METRICS
# =========================================================

def persist_metrics(metrics_dataframe):

    """
    Persist evaluation metrics.
    """

    metrics_dataframe.to_csv(

        METRICS_OUTPUT_PATH,

        index=False
    )

    print("\n[INFO] Metrics persisted successfully.")

# =========================================================
# PERSIST THRESHOLD ANALYSIS
# =========================================================

def persist_threshold_analysis(

    threshold_dataframe
):

    """
    Persist threshold analysis.
    """

    threshold_dataframe.to_csv(

        THRESHOLD_OUTPUT_PATH,

        index=False
    )

    print("[INFO] Threshold analysis persisted successfully.")

# =========================================================
# DISPLAY FINAL COMPARISON
# =========================================================

def display_final_comparison(

    metrics_dataframe
):

    """
    Display final model comparison.
    """

    print("\n========================================")
    print("FINAL MODEL COMPARISON")
    print("========================================")

    print(metrics_dataframe)

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    """
    Execute evaluation pipeline.
    """

    print("\n========================================")
    print("MODEL EVALUATION PIPELINE")
    print("========================================")

    create_directories()

    X_train, X_test, y_train, y_test = (

        load_train_test_split()
    )

    models = load_models()

    metrics_results = []

    threshold_results = []

    for model_name, model in models.items():

        metrics = compute_core_metrics(

            model,
            model_name,

            X_test,
            y_test
        )

        metrics_results.append(metrics)

        display_confusion_matrix(

            model,
            model_name,

            X_test,
            y_test
        )

        display_classification_report(

            model,
            model_name,

            X_test,
            y_test
        )

        threshold_df = threshold_analysis(

            model,
            model_name,

            X_test,
            y_test
        )

        threshold_results.append(
            threshold_df
        )

    metrics_dataframe = pd.DataFrame(
        metrics_results
    )

    threshold_dataframe = pd.concat(

        threshold_results,

        ignore_index=True
    )

    persist_metrics(
        metrics_dataframe
    )

    persist_threshold_analysis(
        threshold_dataframe
    )

    display_final_comparison(
        metrics_dataframe
    )

    print("\n[INFO] Model evaluation pipeline completed.")

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()