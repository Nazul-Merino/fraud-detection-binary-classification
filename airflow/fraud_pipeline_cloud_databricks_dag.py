# =========================================================
# FILE: fraud_pipeline_databricks.py
# =========================================================

# Objective:
# Trigger the validated Azure Databricks fraud detection Job
# from Apache Airflow without executing pipeline modules locally.

from datetime import datetime
from datetime import timedelta

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator


default_args = {
    "owner": "nazul",
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="fraud_detection_databricks_job",
    description=(
        "Trigger Azure Databricks Job for the fraud detection "
        "machine learning pipeline"
    ),
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=[
        "fraud-detection",
        "machine-learning",
        "databricks",
        "airflow",
        "cloud-orchestration",
    ],
) as dag:

    run_databricks_fraud_job = DatabricksRunNowOperator(
        task_id="run_databricks_fraud_pipeline",
        databricks_conn_id="databricks_default",
        job_id=1035676258022374,
    )
