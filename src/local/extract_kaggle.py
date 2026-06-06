# =========================================================
# FILE: extract_kaggle.py
# =========================================================

# Objective: Download the fraud detection dataset from Kaggle, extract compressed files, and prepare the raw ingestion layer for downstream pipeline stages.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import os
import zipfile
import subprocess

from dotenv import load_dotenv

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# CONFIGURE KAGGLE API TOKEN
# =========================================================

KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

if not KAGGLE_API_TOKEN:

    raise ValueError(
        "[ERROR] Missing KAGGLE_API_TOKEN "
        "environment variable."
    )

os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

# =========================================================
# DEFINE DATASET
# =========================================================

DATASET = "mlg-ulb/creditcardfraud"

# =========================================================
# DEFINE LOCAL STORAGE PATHS
# =========================================================

RAW_DATA_DIR = "data/raw"

ZIP_FILE_NAME = "creditcardfraud.zip"

ZIP_FILE_PATH = os.path.join(
    RAW_DATA_DIR,
    ZIP_FILE_NAME
)

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():

    """
    Create required local directories
    for raw dataset persistence.
    """

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print(f"[INFO] Directory verified: {RAW_DATA_DIR}")

# =========================================================
# DOWNLOAD DATASET FROM KAGGLE
# =========================================================

def download_dataset():

    """
    Download dataset from Kaggle API.
    """

    print("[INFO] Starting Kaggle dataset download...")

    command = [

        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        RAW_DATA_DIR
    ]

    subprocess.run(command, check=True)

    print("[INFO] Dataset downloaded successfully.")

# =========================================================
# EXTRACT ZIP FILE
# =========================================================

def extract_dataset():

    """
    Extract downloaded ZIP file.
    """

    if not os.path.exists(ZIP_FILE_PATH):

        raise FileNotFoundError(
            f"[ERROR] ZIP file not found: "
            f"{ZIP_FILE_PATH}"
        )

    print("[INFO] Extracting dataset...")

    with zipfile.ZipFile(ZIP_FILE_PATH, "r") as zip_ref:

        zip_ref.extractall(RAW_DATA_DIR)

    print("[INFO] Dataset extracted successfully.")

# =========================================================
# VALIDATE EXTRACTED FILES
# =========================================================

def validate_extraction():

    """
    Display extracted files
    for validation purposes.
    """

    extracted_files = os.listdir(RAW_DATA_DIR)

    print("\n[INFO] Available files:")

    for file in extracted_files:

        print(f" - {file}")

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    """
    Execute full extraction workflow.
    """

    print("\n=================================================")
    print("KAGGLE DATA EXTRACTION PIPELINE")
    print("=================================================\n")

    create_directories()

    download_dataset()

    extract_dataset()

    validate_extraction()

    print("\n[INFO] Raw dataset ingestion completed.")

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()

# =========================================================
# End of "extract_kaggle.py"
# =========================================================