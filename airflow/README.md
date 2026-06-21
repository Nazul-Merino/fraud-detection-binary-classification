# Apache Airflow Orchestration

This folder contains the Apache Airflow DAGs used to orchestrate the fraud detection pipeline in both local and cloud environments.

## DAGs

### fraud_pipeline_local_dag.py

Implements the local orchestration workflow using Apache Airflow and BashOperator tasks. The DAG executes each pipeline stage sequentially:

1. Data extraction from Kaggle
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation

This version was used to validate the complete machine learning workflow in a local execution environment.

---

### fraud_pipeline_cloud_databricks_dag.py

Implements the cloud orchestration workflow using Apache Airflow and Azure Databricks integration.

Instead of executing individual Python scripts locally, this DAG triggers a previously configured Azure Databricks Job that executes the complete distributed PySpark pipeline in the cloud. The Databricks Job contains the following stages:

1. Data extraction
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation

This architecture follows a production-oriented approach where:

* Apache Airflow acts as the orchestration layer.
* Azure Databricks acts as the distributed processing engine.
* Azure Data Lake Storage (ADLS) acts as the centralized storage layer.

## Orchestration Evidence

Implementation screenshots are available in the `screenshots/` folder:

* **1_airflow_local_pipeline.png**
  Successful execution of the local Apache Airflow pipeline showing the complete DAG with all tasks completed.

* **2_databricks_cloud_job.png**
  Azure Databricks Job implementation showing the cloud-native orchestration workflow and task dependencies.

* **3_airflow_databricks_orchestration.png**
  Successful integration between Apache Airflow and Azure Databricks, demonstrating Airflow triggering the Databricks Job and orchestrating the complete cloud pipeline.

## Architecture Evolution

This project intentionally includes both orchestration approaches to demonstrate the transition from a local machine learning workflow to a cloud-oriented, distributed data platform architecture.

Local Architecture:

Airflow → Python Modules

Cloud Architecture:

Airflow → Azure Databricks Job → PySpark Pipeline → ADLS

