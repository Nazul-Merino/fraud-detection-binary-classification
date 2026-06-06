# =========================================================
# FILE: preprocess.py
# =========================================================

# Objective: Load raw fraud detection dataset from ADLS, apply preprocessing logic using PySpark, and generate processed outputs for downstream stages.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log

# =========================================================
# DEFINE ADLS PATHS
# =========================================================

RAW_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/raw/creditcard_raw"
)

PROCESSED_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/processed/creditcard_processed"
)

# =========================================================
# CREATE SPARK SESSION
# =========================================================

def create_spark_session():

    print(
        "[INFO] Initializing Spark session..."
    )

    spark = (
        SparkSession
        .builder
        .appName(
            "Fraud_Preprocessing"
        )
        .getOrCreate()
    )

    return spark

# =========================================================
# LOAD RAW DATASET
# =========================================================

def load_raw_dataset(spark):

    print(
        "[INFO] Loading raw dataset..."
    )

    dataframe = spark.read.parquet(
        RAW_DATA_PATH
    )

    print(
        "[INFO] Raw dataset loaded successfully."
    )

    return dataframe

# =========================================================
# VALIDATE DATASET STRUCTURE
# =========================================================

def validate_dataset(dataframe):

    print("\n========================================")
    print("DATASET VALIDATION")
    print("========================================")

    row_count = dataframe.count()

    column_count = len(
        dataframe.columns
    )

    print(
        f"Rows    : {row_count}"
    )

    print(
        f"Columns : {column_count}"
    )

# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(dataframe):

    print(
        "\n[INFO] Removing duplicated rows..."
    )

    initial_count = (
        dataframe.count()
    )

    dataframe = (
        dataframe.dropDuplicates()
    )

    final_count = (
        dataframe.count()
    )

    removed_rows = (
        initial_count - final_count
    )

    print(
        f"[INFO] Removed duplicated rows: "
        f"{removed_rows}"
    )

    return dataframe

# =========================================================
# CREATE LOG_AMOUNT FEATURE
# =========================================================

def create_log_amount_feature(dataframe):

    print(
        "\n[INFO] Creating Log_Amount feature..."
    )

    dataframe = dataframe.withColumn(

        "Log_Amount",

        log(
            col("Amount") + 1
        )
    )

    print(
        "[INFO] Log_Amount feature created successfully."
    )

    return dataframe

# =========================================================
# VALIDATE NULL VALUES
# =========================================================

def validate_nulls(dataframe):

    print("\n========================================")
    print("NULL VALUE VALIDATION")
    print("========================================")

    for column_name in dataframe.columns:

        count_nulls = (
            dataframe
            .filter(
                col(column_name).isNull()
            )
            .count()
        )

        print(
            f"{column_name}: "
            f"{count_nulls}"
        )

# =========================================================
# WRITE PROCESSED DATASET
# =========================================================

def write_processed_dataset(dataframe):

    print(
        "\n[INFO] Writing processed dataset..."
    )

    (
        dataframe.write
        .mode("overwrite")
        .parquet(
            PROCESSED_DATA_PATH
        )
    )

    print(
        "[INFO] Processed dataset written successfully."
    )

# =========================================================
# DISPLAY FINAL DATASET INFORMATION
# =========================================================

def display_final_dataset_info(dataframe):

    print("\n========================================")
    print("FINAL DATASET INFORMATION")
    print("========================================")

    print(
        "\nDataset Schema:\n"
    )

    dataframe.printSchema()

    print(
        "\n[INFO] Sample rows:\n"
    )

    dataframe.show(
        5,
        truncate=False
    )

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    print("\n========================================")
    print("PREPROCESSING PIPELINE")
    print("========================================")

    spark = create_spark_session()

    dataframe = load_raw_dataset(
        spark
    )

    validate_dataset(
        dataframe
    )

    dataframe = remove_duplicates(
        dataframe
    )

    dataframe = create_log_amount_feature(
        dataframe
    )

    validate_nulls(
        dataframe
    )

    write_processed_dataset(
        dataframe
    )

    display_final_dataset_info(
        dataframe
    )

    print(
        "\n[INFO] Preprocessing pipeline completed."
    )

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

main()

# =========================================================
# End of "preprocess.py"
# =========================================================