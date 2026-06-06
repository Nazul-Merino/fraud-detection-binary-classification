# =========================================================
# FILE: evaluate_model.py
# =========================================================

# Objective: Load trained Spark ML fraud detection models, evaluate model performance, perform threshold analysis, compare models,
# and persist evaluation artifacts in ADLS.

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np

from sklearn.metrics import (

    average_precision_score,
    precision_score,
    recall_score,
    f1_score,

    confusion_matrix,
    classification_report
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.functions import (
    vector_to_array
)

from pyspark.ml.classification import (

    LogisticRegressionModel,

    RandomForestClassificationModel
)

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator
)

# =========================================================
# DEFINE ADLS PATHS
# =========================================================

TEST_DATA_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/test/creditcard_test"
)

MODEL_METRICS_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/evaluation/model_metrics"
)

THRESHOLD_ANALYSIS_PATH = (
    "abfss://raw-data@datalake123xyz.dfs.core.windows.net/"
    "fraud/evaluation/threshold_analysis"
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
            "Fraud_Model_Evaluation"
        )
        .getOrCreate()
    )

    return spark

# =========================================================
# LOAD TEST DATASET
# =========================================================

def load_test_dataset(spark):

    print(
        "[INFO] Loading test dataset..."
    )

    dataframe = spark.read.parquet(
        TEST_DATA_PATH
    )

    print(
        "[INFO] Test dataset loaded successfully."
    )

    return dataframe

# =========================================================
# LOAD MODELS
# =========================================================

def load_models():

    print(
        "[INFO] Loading trained models..."
    )

    models = {

        "Logistic Regression":

        LogisticRegressionModel.load(
            LOGISTIC_MODEL_PATH
        ),

        "Random Forest":

        RandomForestClassificationModel.load(
            RANDOM_FOREST_MODEL_PATH
        )
    }

    print(
        "[INFO] Models loaded successfully."
    )

    return models

# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def generate_predictions(

    model,
    test_df
):

    predictions = model.transform(
        test_df
    )

    predictions = (

        predictions

        .withColumn(

            "fraud_probability",

            vector_to_array(

                F.col(
                    "probability"
                )

            )[1]
        )
    )

    return predictions

# =========================================================
# CONVERT TO PANDAS
# =========================================================

def convert_predictions_to_pandas(

    predictions
):

    pdf = (

        predictions

        .select(

            "Class",

            "prediction",

            "fraud_probability"
        )

        .toPandas()
    )

    return pdf

# =========================================================
# COMPUTE CORE METRICS
# =========================================================

def compute_core_metrics(

    predictions,
    pdf,
    model_name
):

    evaluator = (

        BinaryClassificationEvaluator(

            labelCol="Class",

            rawPredictionCol="rawPrediction",

            metricName="areaUnderROC"
        )
    )

    roc_auc = evaluator.evaluate(
        predictions
    )

    pr_auc = average_precision_score(

        pdf["Class"],

        pdf["fraud_probability"]
    )

    precision = precision_score(

        pdf["Class"],

        pdf["prediction"],

        zero_division=0
    )

    recall = recall_score(

        pdf["Class"],

        pdf["prediction"],

        zero_division=0
    )

    f1 = f1_score(

        pdf["Class"],

        pdf["prediction"],

        zero_division=0
    )

    metrics = {

        "Model": model_name,

        "PR_AUC": pr_auc,

        "ROC_AUC": roc_auc,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1
    }

    return metrics

# =========================================================
# DISPLAY CONFUSION MATRIX
# =========================================================

def display_confusion_matrix(

    pdf,
    model_name
):

    matrix = confusion_matrix(

        pdf["Class"],

        pdf["prediction"]
    )

    print("\n========================================")
    print(
        f"{model_name.upper()} CONFUSION MATRIX"
    )
    print("========================================")

    print(matrix)

# =========================================================
# DISPLAY CLASSIFICATION REPORT
# =========================================================

def display_classification_report(

    pdf,
    model_name
):

    report = classification_report(

        pdf["Class"],

        pdf["prediction"],

        zero_division=0
    )

    print("\n========================================")
    print(
        f"{model_name.upper()} CLASSIFICATION REPORT"
    )
    print("========================================")

    print(report)

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

def threshold_analysis(

    pdf,
    model_name
):

    threshold_results = []

    thresholds = np.arange(

        0.10,
        0.91,
        0.10
    )

    for threshold in thresholds:

        predictions = (

            pdf[
                "fraud_probability"
            ] >= threshold

        ).astype(int)

        precision = precision_score(

            pdf["Class"],

            predictions,

            zero_division=0
        )

        recall = recall_score(

            pdf["Class"],

            predictions,

            zero_division=0
        )

        f1 = f1_score(

            pdf["Class"],

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

def persist_metrics(

    spark,
    metrics_dataframe
):

    print(
        "\n[INFO] Persisting model metrics..."
    )

    spark_df = spark.createDataFrame(
        metrics_dataframe
    )

    (
        spark_df.write
        .mode("overwrite")
        .parquet(
            MODEL_METRICS_PATH
        )
    )

    print(
        "[INFO] Model metrics persisted."
    )

# =========================================================
# PERSIST THRESHOLD ANALYSIS
# =========================================================

def persist_threshold_analysis(

    spark,
    threshold_dataframe
):

    print(
        "\n[INFO] Persisting threshold analysis..."
    )

    spark_df = spark.createDataFrame(
        threshold_dataframe
    )

    (
        spark_df.write
        .mode("overwrite")
        .parquet(
            THRESHOLD_ANALYSIS_PATH
        )
    )

    print(
        "[INFO] Threshold analysis persisted."
    )

# =========================================================
# DISPLAY FINAL COMPARISON
# =========================================================

def display_final_comparison(

    metrics_dataframe
):

    print("\n========================================")
    print("FINAL MODEL COMPARISON")
    print("========================================")

    print(
        metrics_dataframe
    )

# =========================================================
# MAIN EXECUTION LOGIC
# =========================================================

def main():

    print("\n========================================")
    print("MODEL EVALUATION PIPELINE")
    print("========================================")

    spark = create_spark_session()

    test_df = load_test_dataset(
        spark
    )

    models = load_models()

    metrics_results = []

    threshold_results = []

    for model_name, model in models.items():

        predictions = generate_predictions(

            model,
            test_df
        )

        pdf = convert_predictions_to_pandas(
            predictions
        )

        metrics = compute_core_metrics(

            predictions,
            pdf,
            model_name
        )

        metrics_results.append(
            metrics
        )

        display_confusion_matrix(

            pdf,
            model_name
        )

        display_classification_report(

            pdf,
            model_name
        )

        threshold_df = threshold_analysis(

            pdf,
            model_name
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

        spark,

        metrics_dataframe
    )

    persist_threshold_analysis(

        spark,

        threshold_dataframe
    )

    display_final_comparison(
        metrics_dataframe
    )

    print(
        "\n[INFO] Model evaluation pipeline completed."
    )

# =========================================================
# SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()

# =========================================================
# End of evaluate_model.py
# =========================================================