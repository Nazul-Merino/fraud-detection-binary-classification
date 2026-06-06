# =========================================================
# FILE: train_model.py
# =========================================================

# Objective: Load engineered fraud detection features from ADLS, create train/test datasets, train Spark ML fraud detection models,
# and persist model artifacts.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)

# =========================================================
# DEFINE ADLS PATHS
# =========================================================

FEATURE_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/features/creditcard_features"
)

TRAIN_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/train/creditcard_train"
)

TEST_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/test/creditcard_test"
)

LOGISTIC_MODEL_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/models/logistic_regression"
)

RANDOM_FOREST_MODEL_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/models/random_forest"
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
            "Fraud_Model_Training"
        )
        .getOrCreate()
    )

    return spark

# =========================================================
# LOAD FEATURE DATASET
# =========================================================

def load_feature_dataset(spark):

    print(
        "[INFO] Loading feature dataset..."
    )

    dataframe = spark.read.parquet(
        FEATURE_DATA_PATH
    )

    print(
        "[INFO] Feature dataset loaded successfully."
    )

    return dataframe

# =========================================================
# VALIDATE DATASET
# =========================================================

def validate_dataset(dataframe):

    print("\n========================================")
    print("MODEL TRAINING DATA VALIDATION")
    print("========================================")

    print(
        f"Rows : {dataframe.count()}"
    )

    print(
        f"Columns : {len(dataframe.columns)}"
    )

# =========================================================
# CREATE WEIGHT COLUMN
# =========================================================

def create_weight_column(dataframe):

    print(
        "\n[INFO] Creating weight column..."
    )

    class_counts = (

        dataframe
        .groupBy("Class")
        .count()
        .collect()
    )

    counts = {

        row["Class"]: row["count"]

        for row in class_counts
    }

    majority_count = counts[0]

    minority_count = counts[1]

    fraud_weight = (

        majority_count
        / minority_count
    )

    dataframe = dataframe.withColumn(

        "weight",

        F.when(

            F.col("Class") == 1,

            fraud_weight

        ).otherwise(1.0)
    )

    print(
        f"[INFO] Fraud weight: "
        f"{fraud_weight:.4f}"
    )

    return dataframe

# =========================================================
# CREATE TRAIN TEST SPLIT
# =========================================================

def create_train_test_split(dataframe):

    print(
        "\n[INFO] Creating train/test split..."
    )

    train_df, test_df = (

        dataframe.randomSplit(

            [0.8, 0.2],

            seed=42
        )
    )

    print(
        "[INFO] Train/test split completed."
    )

    print(
        f"[INFO] Train rows : "
        f"{train_df.count()}"
    )

    print(
        f"[INFO] Test rows : "
        f"{test_df.count()}"
    )

    return train_df, test_df

# =========================================================
# PERSIST TRAIN TEST DATASETS
# =========================================================

def persist_train_test_datasets(

    train_df,
    test_df
):

    print(
        "\n[INFO] Persisting train dataset..."
    )

    (
        train_df.write
        .mode("overwrite")
        .parquet(
            TRAIN_DATA_PATH
        )
    )

    print(
        "[INFO] Train dataset persisted."
    )

    print(
        "\n[INFO] Persisting test dataset..."
    )

    (
        test_df.write
        .mode("overwrite")
        .parquet(
            TEST_DATA_PATH
        )
    )

    print(
        "[INFO] Test dataset persisted."
    )

# =========================================================
# TRAIN LOGISTIC REGRESSION
# =========================================================

def train_logistic_regression(

    train_df
):

    print(
        "\n[INFO] Training Logistic Regression..."
    )

    model = LogisticRegression(

        featuresCol="features",

        labelCol="Class",

        weightCol="weight",

        maxIter=100
    )

    fitted_model = model.fit(
        train_df
    )

    print(
        "[INFO] Logistic Regression training completed."
    )

    return fitted_model

# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

def train_random_forest(

    train_df
):

    print(
        "\n[INFO] Training Random Forest..."
    )

    model = RandomForestClassifier(

        featuresCol="features",

        labelCol="Class",

        weightCol="weight",

        numTrees=100,

        seed=42
    )

    fitted_model = model.fit(
        train_df
    )

    print(
        "[INFO] Random Forest training completed."
    )

    return fitted_model

# =========================================================
# PERSIST MODELS
# =========================================================

def persist_models(

    logistic_model,
    random_forest_model
):

    print(
        "\n[INFO] Persisting Logistic Regression..."
    )

    (
        logistic_model.write()
        .overwrite()
        .save(
            LOGISTIC_MODEL_PATH
        )
    )

    print(
        "[INFO] Logistic Regression persisted."
    )

    print(
        "\n[INFO] Persisting Random Forest..."
    )

    (
        random_forest_model.write()
        .overwrite()
        .save(
            RANDOM_FOREST_MODEL_PATH
        )
    )

    print(
        "[INFO] Random Forest persisted."
    )

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    print("\n========================================")
    print("MODEL TRAINING PIPELINE")
    print("========================================")

    spark = create_spark_session()

    dataframe = load_feature_dataset(
        spark
    )

    validate_dataset(
        dataframe
    )

    dataframe = create_weight_column(
        dataframe
    )

    train_df, test_df = (

        create_train_test_split(
            dataframe
        )
    )

    persist_train_test_datasets(

        train_df,
        test_df
    )

    logistic_model = (

        train_logistic_regression(
            train_df
        )
    )

    random_forest_model = (

        train_random_forest(
            train_df
        )
    )

    persist_models(

        logistic_model,

        random_forest_model
    )

    print(
        "\n[INFO] Model training pipeline completed."
    )

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

main()

# =========================================================
# End of train_model.py
# =========================================================