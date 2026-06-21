# Infrastructure Screenshots

This directory contains screenshots and architecture diagrams that provide visual evidence of the implementation and successful execution of both the local and cloud-based fraud detection pipelines.

## Contents

### 1_airflow_local_pipeline.png

Screenshot of the Apache Airflow local orchestration workflow. The image demonstrates the successful execution of the end-to-end fraud detection pipeline, including data extraction, preprocessing, feature engineering, model training, and model evaluation tasks coordinated through Airflow.

### 2_databricks_cloud_job.png

Screenshot of the Azure Databricks Job used to execute the cloud-based fraud detection workflow. The image shows the task structure and execution dependencies configured within the Databricks environment.

### 3_airflow_databricks_orchestration.png

Screenshot of the Apache Airflow DAG responsible for triggering the Azure Databricks Job. This image demonstrates the integration between Airflow and Databricks, where Airflow acts as the orchestration layer for cloud pipeline execution.

### cloud_pipeline_architecture_scheme.png

Architecture diagram of the cloud implementation. The diagram illustrates the interaction between Kaggle data ingestion, Azure Data Lake Storage (ADLS), Azure Databricks, Apache Airflow, and the machine learning workflow components used throughout the project.

## Purpose

These screenshots complement the source code and technical documentation contained in the repository by providing visual evidence of the infrastructure, orchestration workflows, and cloud architecture implemented for the fraud detection project.

