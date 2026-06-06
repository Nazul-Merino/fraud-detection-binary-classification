# =========================================================
# FILE: feature_engineering.py
# =========================================================

# Objective: Load the processed fraud detection dataset, apply finalized feature engineering logic, and generate model-ready feature datasets
# for downstream modeling stages.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os

import joblib
import pandas as pd

from sklearn.preprocessing import RobustScaler

# =========================================================
# DEFINE LOCAL DATA PATHS
# =========================================================

PROCESSED_DATA_PATH = (
    "data/processed/creditcard_processed.parquet"
)

FEATURE_OUTPUT_PATH = (
    "data/features/creditcard_features.parquet"
)

SCALER_OUTPUT_PATH = (
    "data/features/robust_scaler.pkl"
)

FEATURE_SCHEMA_PATH = (
    "data/features/feature_columns.pkl"
)

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():

    """
    Create required local directories
    for feature persistence.
    """

    os.makedirs("data/features", exist_ok=True)

    print("[INFO] Feature directory verified.")

# =========================================================
# LOAD PROCESSED DATASET
# =========================================================

def load_processed_dataset():

    """
    Load processed parquet dataset
    into pandas DataFrame.
    """

    print("[INFO] Loading processed dataset...")

    dataframe = pd.read_parquet(
        PROCESSED_DATA_PATH
    )

    print("[INFO] Processed dataset loaded successfully.")

    return dataframe

# =========================================================
# VALIDATE DATASET STRUCTURE
# =========================================================

def validate_dataset(dataframe):

    """
    Display dataset validation information.
    """

    print("\n========================================")
    print("FEATURE DATASET VALIDATION")
    print("========================================")

    print(f"Rows    : {dataframe.shape[0]}")

    print(f"Columns : {dataframe.shape[1]}")

# =========================================================
# DEFINE FEATURE COLUMNS
# =========================================================

def define_feature_columns(dataframe):

    """
    Define model feature columns
    excluding target variable.
    """

    feature_columns = [

        column
        for column in dataframe.columns
        if column != "Class"
    ]

    print("\n[INFO] Feature columns defined successfully.")

    print(f"[INFO] Total feature columns: "
          f"{len(feature_columns)}")

    return feature_columns

# =========================================================
# APPLY ROBUST SCALING
# =========================================================

def apply_robust_scaling(
    dataframe,
    feature_columns
):

    """
    Apply RobustScaler based on
    Phase 1 and Phase 2 findings.
    """

    print("\n[INFO] Applying RobustScaler...")

    scaler = RobustScaler()

    dataframe[feature_columns] = scaler.fit_transform(

        dataframe[feature_columns]
    )

    print("[INFO] Robust scaling completed successfully.")

    return dataframe, scaler

# =========================================================
# PERSIST SCALER ARTIFACT
# =========================================================

def persist_scaler(scaler):

    """
    Persist fitted scaler artifact
    for reproducible inference.
    """

    joblib.dump(
        scaler,
        SCALER_OUTPUT_PATH
    )

    print("[INFO] RobustScaler artifact persisted.")

# =========================================================
# PERSIST FEATURE SCHEMA
# =========================================================

def persist_feature_schema(feature_columns):

    """
    Persist feature column schema
    for reproducible inference.
    """

    joblib.dump(
        feature_columns,
        FEATURE_SCHEMA_PATH
    )

    print("[INFO] Feature schema persisted.")

# =========================================================
# WRITE FEATURE DATASET
# =========================================================

def write_feature_dataset(dataframe):

    """
    Persist engineered feature dataset
    in parquet format.
    """

    print("\n[INFO] Writing feature dataset...")

    dataframe.to_parquet(

        FEATURE_OUTPUT_PATH,

        index=False
    )

    print("[INFO] Feature dataset written successfully.")

# =========================================================
# DISPLAY FINAL FEATURE INFORMATION
# =========================================================

def display_final_feature_info(dataframe):

    """
    Display final feature dataset information.
    """

    print("\n========================================")
    print("FINAL FEATURE DATASET INFORMATION")
    print("========================================")

    print("\nDataset Schema:\n")

    print(dataframe.dtypes)

    print("\n[INFO] Sample rows:\n")

    print(dataframe.head())

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    """
    Execute feature engineering pipeline.
    """

    print("\n========================================")
    print("FEATURE ENGINEERING PIPELINE")
    print("========================================")

    create_directories()

    dataframe = load_processed_dataset()

    validate_dataset(dataframe)

    feature_columns = define_feature_columns(
        dataframe
    )

    dataframe, scaler = apply_robust_scaling(

        dataframe,

        feature_columns
    )

    persist_scaler(scaler)

    persist_feature_schema(feature_columns)

    write_feature_dataset(dataframe)

    display_final_feature_info(dataframe)

    print("\n[INFO] Feature engineering pipeline completed.")

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()