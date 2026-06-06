# =========================================================
# FILE: preprocess.py
# =========================================================

# Objective: Load the raw fraud detection dataset, apply finalized preprocessing logic, and generate processed parquet outputs
# for downstream pipeline stages.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os

import numpy as np
import pandas as pd

# =========================================================
# DEFINE LOCAL DATA PATHS
# =========================================================

RAW_DATA_PATH = "data/raw/creditcard.csv"

PROCESSED_OUTPUT_PATH = (
    "data/processed/creditcard_processed.parquet"
)

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():

    """
    Create required local directories
    for processed datasets.
    """

    os.makedirs("data/processed", exist_ok=True)

    print("[INFO] Processed data directory verified.")

# =========================================================
# LOAD RAW DATASET
# =========================================================

def load_raw_dataset():

    """
    Load raw CSV dataset into pandas DataFrame.
    """

    print("[INFO] Loading raw dataset...")

    dataframe = pd.read_csv(RAW_DATA_PATH)

    print("[INFO] Raw dataset loaded successfully.")

    return dataframe

# =========================================================
# VALIDATE DATASET STRUCTURE
# =========================================================

def validate_dataset(dataframe):

    """
    Display dataset validation information.
    """

    print("\n========================================")
    print("DATASET VALIDATION")
    print("========================================")

    print(f"Rows    : {dataframe.shape[0]}")

    print(f"Columns : {dataframe.shape[1]}")

# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(dataframe):

    """
    Remove duplicated rows
    identified during Phase 1 analysis.
    """

    print("\n[INFO] Removing duplicated rows...")

    initial_count = dataframe.shape[0]

    dataframe = dataframe.drop_duplicates()

    final_count = dataframe.shape[0]

    removed_rows = initial_count - final_count

    print(f"[INFO] Removed duplicated rows: {removed_rows}")

    return dataframe

# =========================================================
# CREATE LOG-TRANSFORMED AMOUNT FEATURE
# =========================================================

def create_log_amount_feature(dataframe):

    """
    Create log-transformed Amount feature
    based on Phase 1 findings.
    """

    print("\n[INFO] Creating Log_Amount feature...")

    dataframe["Log_Amount"] = np.log1p(
        dataframe["Amount"]
    )

    print("[INFO] Log_Amount feature created successfully.")

    return dataframe

# =========================================================
# VALIDATE NULL VALUES
# =========================================================

def validate_nulls(dataframe):

    """
    Validate null values across columns.
    """

    print("\n========================================")
    print("NULL VALUE VALIDATION")
    print("========================================")

    null_counts = dataframe.isnull().sum()

    print(null_counts)

# =========================================================
# WRITE PROCESSED PARQUET DATASET
# =========================================================

def write_processed_dataset(dataframe):

    """
    Persist processed dataset
    in parquet format.
    """

    print("\n[INFO] Writing processed parquet dataset...")

    dataframe.to_parquet(

        PROCESSED_OUTPUT_PATH,

        index=False
    )

    print("[INFO] Processed parquet dataset written successfully.")

# =========================================================
# DISPLAY FINAL DATASET INFORMATION
# =========================================================

def display_final_dataset_info(dataframe):

    """
    Display final dataset structure.
    """

    print("\n========================================")
    print("FINAL DATASET INFORMATION")
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
    Execute preprocessing pipeline.
    """

    print("\n========================================")
    print("PREPROCESSING PIPELINE")
    print("========================================")

    create_directories()

    dataframe = load_raw_dataset()

    validate_dataset(dataframe)

    dataframe = remove_duplicates(dataframe)

    dataframe = create_log_amount_feature(dataframe)

    validate_nulls(dataframe)

    write_processed_dataset(dataframe)

    display_final_dataset_info(dataframe)

    print("\n[INFO] Preprocessing pipeline completed.")

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()