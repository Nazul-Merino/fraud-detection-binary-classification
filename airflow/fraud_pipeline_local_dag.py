# =========================================================
# FILE: fraud_pipeline_dag.py
# =========================================================

# Objective:
# Orchestrate the complete local fraud detection
# machine learning pipeline using Apache Airflow.
#
# Current execution mode:
# - Local modular execution
#
# Future evolution:
# - Databricks orchestration
# - ADLS persistence
# - distributed execution

# =========================================================
# IMPORT LIBRARIES
# =========================================================

from airflow import DAG

from airflow.operators.bash import BashOperator

from datetime import datetime
from datetime import timedelta

# =========================================================
# DEFAULT DAG CONFIGURATION
# =========================================================

default_args = {

    "owner": "nazul",

    "depends_on_past": False,

    "start_date": datetime(2026, 6, 1),

    "email_on_failure": False,

    "email_on_retry": False,

    "retries": 1,

    "retry_delay": timedelta(minutes=5)
}

# =========================================================
# CREATE DAG
# =========================================================

with DAG(

    dag_id="fraud_detection_pipeline",

    description=(
        "Local modular fraud detection "
        "ML pipeline orchestration"
    ),

    default_args=default_args,

    schedule_interval=None,

    catchup=False,

    tags=[
        "fraud-detection",
        "machine-learning",
        "local-validation",
        "pipeline"
    ]

) as dag:

    # =====================================================
    # TASK 1 — EXTRACT KAGGLE DATA
    # =====================================================

    extract_kaggle_task = BashOperator(

        task_id="extract_kaggle_data",

        bash_command=(
            "cd /mnt/c/projects/portafolio_de_proyectos && "
            "python3 extract_kaggle.py"
        )
    )

    # =====================================================
    # TASK 2 — PREPROCESS DATA
    # =====================================================

    preprocess_task = BashOperator(

        task_id="preprocess_data",

        bash_command=(
            "cd /mnt/c/projects/portafolio_de_proyectos && "
            "python3 preprocess.py"
        )
    )

    # =====================================================
    # TASK 3 — FEATURE ENGINEERING
    # =====================================================

    feature_engineering_task = BashOperator(

        task_id="feature_engineering",

        bash_command=(
            "cd /mnt/c/projects/portafolio_de_proyectos && "
            "python3 feature_engineering.py"
        )
    )

    # =====================================================
    # TASK 4 — TRAIN MODELS
    # =====================================================

    train_model_task = BashOperator(

        task_id="train_models",

        bash_command=(
            "cd /mnt/c/projects/portafolio_de_proyectos && "
            "python3 train_model.py"
        )
    )

    # =====================================================
    # TASK 5 — EVALUATE MODELS
    # =====================================================

    evaluate_model_task = BashOperator(

        task_id="evaluate_models",

        bash_command=(
            "cd /mnt/c/projects/portafolio_de_proyectos && "
            "python3 evaluate_model.py"
        )
    )

    # =====================================================
    # DEFINE TASK DEPENDENCIES
    # =====================================================

    (
        extract_kaggle_task

        >> preprocess_task

        >> feature_engineering_task

        >> train_model_task

        >> evaluate_model_task
    )
