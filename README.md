# Credit Card Fraud Detection Pipeline

## Problem

Financial institutions process millions of transactions every day and face significant financial losses due to fraudulent activity. Detecting fraudulent transactions is particularly challenging because fraud events are extremely rare compared to legitimate transactions, resulting in highly imbalanced datasets.

Traditional evaluation metrics such as Accuracy may be misleading in this context, making it necessary to focus on metrics that properly capture fraud detection performance and operational trade-offs.

---

## Objective

Develop an end-to-end fraud detection pipeline capable of identifying fraudulent credit card transactions using machine learning techniques while following production-oriented engineering practices.

The project covers the complete workflow from data ingestion and exploratory analysis to cloud deployment, orchestration, model training, evaluation, and artifact persistence.

---

## Dataset Overview

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle:

* Source: Kaggle
* Dataset: Credit Card Fraud Detection
* Records: 284,807 transactions
* Variables: 31
* Target variable: `Class`

  * `0` = Legitimate transaction
  * `1` = Fraudulent transaction
* Fraud cases: 492
* Fraud rate: approximately 0.17%
* Class imbalance ratio: approximately 578:1

The dataset includes 28 anonymized variables (`V1`–`V28`) generated through Principal Component Analysis (PCA), along with transaction amount and transaction time information.

---

## Approach

The project was developed in two major phases.

### Phase 1 — Exploratory Analysis and Model Development

* Data quality assessment
* Missing value analysis
* Duplicate detection
* Class imbalance analysis
* Exploratory data analysis (EDA)
* Feature engineering experiments
* Model development and comparison
* Threshold optimization analysis
* Selection of production-oriented modeling strategy

### Phase 2 — Production Architecture and Cloud Deployment

* Migration from notebook-based experimentation to modular production scripts
* PySpark implementation
* Azure Databricks integration
* Azure Data Lake Storage (ADLS Gen2) persistence
* Apache Airflow orchestration
* Automated dataset ingestion through Kaggle API
* End-to-end cloud pipeline validation

The exploratory analysis, experimentation, and model development process can be found in the `notebooks/` directory.

Readers interested in the complete technical findings, model evaluation results, and architecture decisions may refer to:

- `docs/phase_1_exploratory_analysis_summary.pdf`
- `docs/phase_2_production_architecture_summary.pdf`

---

## Handling Extreme Class Imbalance

Fraudulent transactions represent only approximately **0.17%** of all observations.

Because of this severe imbalance, model evaluation focused primarily on:

* Precision
* Recall
* F1-Score
* Precision-Recall AUC (PR-AUC)
* False Positive Rate
* False Negative Rate

rather than relying exclusively on Accuracy or ROC-AUC.

This evaluation strategy better reflects the operational requirements of real-world fraud detection systems.

---

## Key Results

* Successfully developed and validated multiple fraud detection models.
* Compared Logistic Regression, Random Forest, XGBoost, and LightGBM.
* Identified XGBoost as the strongest model from a purely predictive perspective.
* Selected Random Forest as the primary production-oriented model due to its strong balance between fraud detection performance, false-positive control, operational stability, and Spark compatibility.
* Successfully implemented and validated a complete cloud-oriented fraud detection architecture.
* Successfully orchestrated the entire pipeline using Apache Airflow and Azure Databricks.

  A comprehensive summary of these analytical results can be found in: docs/fraud_detection_results_showcase.pdf
---

## Technology Stack

### Data Science & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM

### Distributed Processing

* PySpark
* Spark ML

### Cloud & Storage

* Azure Databricks
* Azure Data Lake Storage Gen2 (ADLS)

### Orchestration

* Apache Airflow

### Data Access

* Kaggle API

### Development & Version Control

* GitHub

---

## Scalability Considerations

Although the current dataset contains approximately 285,000 transactions, the project was intentionally designed following cloud-native and distributed-processing principles.

The architecture leverages:

* PySpark
* Azure Databricks
* Azure Data Lake Storage Gen2
* Modular production scripts
* Apache Airflow orchestration

As a result, the current implementation can evolve toward larger transactional workloads with appropriate infrastructure sizing, cluster tuning, and storage optimization.

The project therefore serves as both a machine learning solution and a scalable cloud-oriented data pipeline.

---

## Local and Cloud Execution Modes

This repository contains two complete implementations of the fraud detection workflow.

### Local Architecture

* Python modules
* Local execution
* Apache Airflow orchestration
* Local artifact generation

### Cloud Architecture

* PySpark modules
* Azure Databricks execution
* Azure Data Lake Storage persistence
* Apache Airflow orchestration
* Automated cloud deployment workflow

Both implementations were successfully validated.

---

## Repository Structure

### `data/`

Dataset documentation, dataset description and access instructions. Raw datasets are not stored in the repository. Data acquisition is performed automatically through the Kaggle API.

### `notebooks/`

Contains exploratory analysis notebooks, data quality assessment, imbalance analysis, feature engineering experiments, model development, model comparison, threshold optimization studies, and architecture-design decisions.

Readers interested in the complete technical findings may refer to:

* `docs/phase_1_exploratory_analysis_summary.pdf`
* `docs/phase_2_production_architecture_summary.pdf`

### `src/`

Contains production-ready Python modules.

* `local/` → Local pipeline implementation
* `databricks/` → Cloud-oriented Databricks implementation

Both folders contain the following modules:

* `extract_kaggle.py`
* `preprocess.py`
* `feature_engineering.py`
* `train_model.py`
* `evaluate_model.py`

### `airflow/`

Contains Apache Airflow DAGs.

* `fraud_pipeline_local_dag.py`
* `fraud_pipeline_cloud_databricks_dag.py`

These DAGs demonstrate both local and cloud orchestration strategies. Screenshots demonstrating the successful implementation and execution of these orchestration workflows are available in the screenshots/ directory.

### `docs/`

Project documentation and technical summaries.

* `Fraud_Detection_Results_Showcase.pdf`
* `phase_1_exploratory_analysis_summary.pdf`
* `phase_2_production_architecture_summary.pdf`

### `screenshots/`

Execution evidence and architecture diagrams.

Includes screenshots showing:

* Successful local Airflow execution
* Azure Databricks Job implementation
* Airflow-to-Databricks orchestration
* Cloud architecture design
