# =========================================================
# FILE: extract_kaggle.py
# =========================================================

import os

os.environ["KAGGLE_USERNAME"] = "nazulmerino"

os.environ["KAGGLE_KEY"] = "KGAT_1e73ead2f37c60060340b303678a9e25"

# =========================================================

import zipfile
import subprocess
import pandas as pd

from pyspark.sql import SparkSession

# =========================================================
# DEFINE DATASET
# =========================================================

DATASET = "mlg-ulb/creditcardfraud"

# =========================================================
# DEFINE STORAGE PATHS
# =========================================================

RAW_DATA_DIR = "/tmp/data/raw"

ZIP_FILE_NAME = "creditcardfraud.zip"

ZIP_FILE_PATH = os.path.join(
    RAW_DATA_DIR,
    ZIP_FILE_NAME
)

CSV_FILE_PATH = os.path.join(
    RAW_DATA_DIR,
    "creditcard.csv"
)

# =========================================================
# DEFINE ADLS OUTPUT PATH
# =========================================================

RAW_DATA_ADLS_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/raw/creditcard_raw"
)

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

def create_directories():

    os.makedirs(
        RAW_DATA_DIR,
        exist_ok=True
    )

    print(
        f"[INFO] Directory verified: {RAW_DATA_DIR}"
    )

# =========================================================
# DOWNLOAD DATASET FROM KAGGLE
# =========================================================

def download_dataset():

    print(
        "[INFO] Starting Kaggle dataset download..."
    )

    command = [

        "kaggle",
        "datasets",
        "download",

        "-d",
        DATASET,

        "-p",
        RAW_DATA_DIR

    ]

    subprocess.run(
        command,
        check=True
    )

    print(
        "[INFO] Dataset downloaded successfully."
    )

# =========================================================
# EXTRACT ZIP FILE
# =========================================================

def extract_dataset():

    if not os.path.exists(
        ZIP_FILE_PATH
    ):

        raise FileNotFoundError(
            f"[ERROR] ZIP file not found: "
            f"{ZIP_FILE_PATH}"
        )

    print(
        "[INFO] Extracting dataset..."
    )

    with zipfile.ZipFile(
        ZIP_FILE_PATH,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            RAW_DATA_DIR
        )

    print(
        "[INFO] Dataset extracted successfully."
    )

# =========================================================
# VALIDATE EXTRACTED FILES
# =========================================================

def validate_extraction():

    extracted_files = os.listdir(
        RAW_DATA_DIR
    )

    print(
        "\n[INFO] Available files:"
    )

    for file in extracted_files:

        print(
            f" - {file}"
        )

# =========================================================
# LOAD DATA INTO ADLS
# =========================================================

def load_to_adls():

    print(
        "\n[INFO] Loading dataset into ADLS..."
    )

    spark = (
        SparkSession
        .builder
        .getOrCreate()
    )

    pandas_df = pd.read_csv(
        CSV_FILE_PATH
    )

    spark_df = spark.createDataFrame(
        pandas_df
    )

    (
        spark_df.write
        .mode("overwrite")
        .parquet(
            RAW_DATA_ADLS_PATH
        )
    )

    print(
        "[INFO] Dataset successfully loaded into ADLS."
    )

    print(
        f"[INFO] ADLS path: "
        f"{RAW_DATA_ADLS_PATH}"
    )

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    print(
        "\n================================================="
    )

    print(
        "KAGGLE DATA EXTRACTION PIPELINE"
    )

    print(
        "=================================================\n"
    )

    create_directories()

    download_dataset()

    extract_dataset()

    validate_extraction()

    load_to_adls()

    print(
        "\n[INFO] Raw dataset ingestion completed."
    )

# =========================================================
# EXECUTE PIPELINE
# =========================================================

main()

# =========================================================
# End of "extract_kaggle.py"
# =========================================================