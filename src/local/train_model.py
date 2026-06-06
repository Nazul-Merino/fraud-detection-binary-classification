# =========================================================
# FILE: train_model.py
# =========================================================

# Objective: Load engineered fraud detection features, perform stratified train/test splitting, train production-oriented fraud detection models,
# and persist reproducible model artifacts for downstream evaluation stages.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

# =========================================================
# DEFINE LOCAL DATA PATHS
# =========================================================

FEATURE_DATA_PATH = (
    "data/features/creditcard_features.parquet"
)

MODEL_OUTPUT_DIR = "models"

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

TRAIN_TEST_SPLIT_PATH = (
    "models/train_test_split.pkl"
)

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():

    """
    Create required directories
    for model persistence.
    """

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("[INFO] Model directory verified.")

# =========================================================
# LOAD FEATURE DATASET
# =========================================================

def load_feature_dataset():

    """
    Load engineered feature dataset.
    """

    print("[INFO] Loading feature dataset...")

    dataframe = pd.read_parquet(
        FEATURE_DATA_PATH
    )

    print("[INFO] Feature dataset loaded successfully.")

    return dataframe

# =========================================================
# VALIDATE DATASET STRUCTURE
# =========================================================

def validate_dataset(dataframe):

    """
    Display dataset validation information.
    """

    print("\n========================================")
    print("MODEL TRAINING DATA VALIDATION")
    print("========================================")

    print(f"Rows    : {dataframe.shape[0]}")

    print(f"Columns : {dataframe.shape[1]}")

# =========================================================
# PREPARE FEATURES AND TARGET
# =========================================================

def prepare_features_and_target(dataframe):

    """
    Separate feature matrix and target variable.
    """

    X = dataframe.drop(columns=["Class"])

    y = dataframe["Class"]

    print("\n[INFO] Features and target prepared.")

    print(f"[INFO] Feature matrix shape: {X.shape}")

    print(f"[INFO] Target vector shape : {y.shape}")

    return X, y

# =========================================================
# CREATE STRATIFIED TRAIN/TEST SPLIT
# =========================================================

def create_train_test_split(X, y):

    """
    Create stratified train/test split
    preserving fraud distribution.
    """

    print("\n[INFO] Creating stratified train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    print("[INFO] Train/test split completed.")

    print(f"[INFO] X_train shape : {X_train.shape}")

    print(f"[INFO] X_test shape  : {X_test.shape}")

    return X_train, X_test, y_train, y_test

# =========================================================
# TRAIN LOGISTIC REGRESSION
# =========================================================

def train_logistic_regression(X_train, y_train):

    """
    Train Logistic Regression model.
    """

    print("\n[INFO] Training Logistic Regression...")

    model = LogisticRegression(

        class_weight="balanced",

        max_iter=1000,

        random_state=42
    )

    model.fit(X_train, y_train)

    print("[INFO] Logistic Regression training completed.")

    return model

# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

def train_random_forest(X_train, y_train):

    """
    Train Random Forest model.
    """

    print("\n[INFO] Training Random Forest...")

    model = RandomForestClassifier(

        n_estimators=100,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("[INFO] Random Forest training completed.")

    return model

# =========================================================
# TRAIN XGBOOST
# =========================================================

def train_xgboost(X_train, y_train):

    """
    Train XGBoost model.
    """

    print("\n[INFO] Training XGBoost...")

    model = XGBClassifier(

        n_estimators=100,

        max_depth=6,

        learning_rate=0.1,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    print("[INFO] XGBoost training completed.")

    return model

# =========================================================
# TRAIN LIGHTGBM
# =========================================================

def train_lightgbm(X_train, y_train):

    """
    Train LightGBM model.
    """

    print("\n[INFO] Training LightGBM...")

    model = LGBMClassifier(

        n_estimators=100,

        learning_rate=0.1,

        random_state=42
    )

    model.fit(X_train, y_train)

    print("[INFO] LightGBM training completed.")

    return model

# =========================================================
# DISPLAY MODEL SCORES
# =========================================================

def display_model_scores(

    model,
    model_name,

    X_test,
    y_test
):

    """
    Display baseline evaluation metrics.
    """

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n========================================")
    print(f"{model_name.upper()} PERFORMANCE")
    print("========================================")

    print(f"PR-AUC  : {pr_auc:.6f}")

    print(f"ROC-AUC : {roc_auc:.6f}")

# =========================================================
# PERSIST TRAINED MODELS
# =========================================================

def persist_models(

    logistic_model,
    random_forest_model,
    xgboost_model,
    lightgbm_model
):

    """
    Persist trained models.
    """

    joblib.dump(

        logistic_model,

        LOGISTIC_MODEL_PATH
    )

    joblib.dump(

        random_forest_model,

        RANDOM_FOREST_MODEL_PATH
    )

    joblib.dump(

        xgboost_model,

        XGBOOST_MODEL_PATH
    )

    joblib.dump(

        lightgbm_model,

        LIGHTGBM_MODEL_PATH
    )

    print("\n[INFO] Model artifacts persisted.")

# =========================================================
# PERSIST TRAIN/TEST SPLIT
# =========================================================

def persist_train_test_split(

    X_train,
    X_test,

    y_train,
    y_test
):

    """
    Persist train/test split
    for reproducible evaluation.
    """

    split_artifacts = {

        "X_train": X_train,
        "X_test": X_test,

        "y_train": y_train,
        "y_test": y_test
    }

    joblib.dump(

        split_artifacts,

        TRAIN_TEST_SPLIT_PATH
    )

    print("[INFO] Train/test split persisted.")

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    """
    Execute model training pipeline.
    """

    print("\n========================================")
    print("MODEL TRAINING PIPELINE")
    print("========================================")

    create_directories()

    dataframe = load_feature_dataset()

    validate_dataset(dataframe)

    X, y = prepare_features_and_target(
        dataframe
    )

    X_train, X_test, y_train, y_test = (

        create_train_test_split(
            X,
            y
        )
    )

    logistic_model = train_logistic_regression(

        X_train,
        y_train
    )

    random_forest_model = train_random_forest(

        X_train,
        y_train
    )

    xgboost_model = train_xgboost(

        X_train,
        y_train
    )

    lightgbm_model = train_lightgbm(

        X_train,
        y_train
    )

    display_model_scores(

        logistic_model,

        "Logistic Regression",

        X_test,
        y_test
    )

    display_model_scores(

        random_forest_model,

        "Random Forest",

        X_test,
        y_test
    )

    display_model_scores(

        xgboost_model,

        "XGBoost",

        X_test,
        y_test
    )

    display_model_scores(

        lightgbm_model,

        "LightGBM",

        X_test,
        y_test
    )

    persist_models(

        logistic_model,

        random_forest_model,

        xgboost_model,

        lightgbm_model
    )

    persist_train_test_split(

        X_train,
        X_test,

        y_train,
        y_test
    )

    print("\n[INFO] Model training pipeline completed.")

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()