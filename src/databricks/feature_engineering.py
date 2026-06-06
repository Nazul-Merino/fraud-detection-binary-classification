# =========================================================
# FILE: feature_engineering.py
# =========================================================

# Objective: Load processed fraud detection dataset from ADLS, apply feature engineering using Spark ML, and generate model-ready feature datasets.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

from pyspark.sql import SparkSession

from pyspark.ml.feature import (
    VectorAssembler,
    RobustScaler
)

# =========================================================
# DEFINE ADLS PATHS
# =========================================================

PROCESSED_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/processed/creditcard_processed"
)

FEATURE_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/features/creditcard_features"
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
            "Fraud_Feature_Engineering"
        )
        .getOrCreate()
    )

    return spark

# =========================================================
# LOAD PROCESSED DATASET
# =========================================================

def load_processed_dataset(spark):

    print(
        "[INFO] Loading processed dataset..."
    )

    dataframe = spark.read.parquet(
        PROCESSED_DATA_PATH
    )

    print(
        "[INFO] Processed dataset loaded successfully."
    )

    return dataframe

# =========================================================
# VALIDATE DATASET STRUCTURE
# =========================================================

def validate_dataset(dataframe):

    print("\n========================================")
    print("FEATURE DATASET VALIDATION")
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
# DEFINE FEATURE COLUMNS
# =========================================================

def define_feature_columns(dataframe):

    feature_columns = [

        column

        for column in dataframe.columns

        if column != "Class"
    ]

    print(
        "\n[INFO] Feature columns defined successfully."
    )

    print(
        f"[INFO] Total feature columns: "
        f"{len(feature_columns)}"
    )

    return feature_columns

# =========================================================
# APPLY ROBUST SCALING
# =========================================================

def apply_robust_scaling(
    dataframe,
    feature_columns
):

    print(
        "\n[INFO] Applying Spark ML RobustScaler..."
    )

    assembler = VectorAssembler(

        inputCols=feature_columns,

        outputCol="features_unscaled"
    )

    assembled_df = assembler.transform(
        dataframe
    )

    scaler = RobustScaler(

        inputCol="features_unscaled",

        outputCol="features"
    )

    scaler_model = scaler.fit(
        assembled_df
    )

    scaled_df = scaler_model.transform(
        assembled_df
    )

    print(
        "[INFO] Robust scaling completed successfully."
    )

    return scaled_df

# =========================================================
# WRITE FEATURE DATASET
# =========================================================

def write_feature_dataset(dataframe):

    print(
        "\n[INFO] Writing feature dataset..."
    )

    (
        dataframe.write
        .mode("overwrite")
        .parquet(
            FEATURE_DATA_PATH
        )
    )

    print(
        "[INFO] Feature dataset written successfully."
    )

# =========================================================
# DISPLAY FINAL FEATURE INFORMATION
# =========================================================

def display_final_feature_info(dataframe):

    print("\n========================================")
    print("FINAL FEATURE DATASET INFORMATION")
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
    print("FEATURE ENGINEERING PIPELINE")
    print("========================================")

    spark = create_spark_session()

    dataframe = load_processed_dataset(
        spark
    )

    validate_dataset(
        dataframe
    )

    feature_columns = define_feature_columns(
        dataframe
    )

    dataframe = apply_robust_scaling(

        dataframe,

        feature_columns
    )

    write_feature_dataset(
        dataframe
    )

    display_final_feature_info(
        dataframe
    )

    print(
        "\n[INFO] Feature engineering pipeline completed."
    )

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

main()

# =========================================================
# End of "feature_engineering.py"
# =========================================================