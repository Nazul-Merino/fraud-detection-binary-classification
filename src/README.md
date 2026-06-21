
# Source Code

This directory contains the production-oriented implementation of the fraud detection pipeline.

The project includes two complete implementations:

* A **local pipeline** designed for development, validation, and reproducibility.
* A **cloud-oriented pipeline** designed using Azure Databricks, PySpark, Azure Data Lake Storage (ADLS Gen2), GitHub integration, and Apache Airflow orchestration.

Both implementations follow the same logical workflow and share the same modular structure.

---

## Directory Structure

```text
src/
├── local/
│   ├── extract_kaggle.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
└── databricks/
    ├── extract_kaggle.py
    ├── preprocess.py
    ├── feature_engineering.py
    ├── train_model.py
    └── evaluate_model.py
```

---

## Common Pipeline Logic

Both implementations follow the same machine learning workflow:

```text
Dataset Extraction
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
```

Both implementations follow the same machine learning workflow and business logic. However, the underlying processing framework differs between environments. The local implementation relies primarily on Pandas and Scikit-Learn, while the cloud-oriented implementation was migrated to PySpark and Spark ML to support distributed processing and cloud scalability. The orchestration strategy also differs between implementations.

---

# Local Pipeline

The local implementation was developed first and served as the initial validation environment for the entire workflow.

The pipeline is orchestrated using Apache Airflow and executes Python modules directly on the local machine.

Workflow:

```text
Apache Airflow
        ↓
extract_kaggle.py
        ↓
preprocess.py
        ↓
feature_engineering.py
        ↓
train_model.py
        ↓
evaluate_model.py
```

The local version is useful for:

* Rapid experimentation
* Pipeline validation
* Development and debugging
* Reproducibility testing

Evidence of successful local orchestration can be found in:

```text
screenshots/1_airflow_local_pipeline.png
```

---

# Cloud-Oriented Pipeline

The cloud implementation represents the production-oriented architecture of the project.

Instead of executing modules directly from Airflow, the workflow leverages Azure Databricks as the execution engine and Azure Data Lake Storage (ADLS Gen2) as the storage layer.

The implementation was migrated from standalone notebooks into modular PySpark production scripts.

---

## GitHub Integration

The Databricks implementation is stored and versioned in GitHub.

To enable cloud execution:

1. The Databricks modules were uploaded to GitHub.
2. A Git Folder integration was configured between Azure Databricks and GitHub.
3. Azure Databricks was configured to read the production scripts directly from the repository.
4. Changes committed to GitHub can be synchronized into Databricks through the Git integration.

This approach provides:

* Version control
* Reproducibility
* Traceability
* Easier maintenance

---

## Azure Databricks Job

A Databricks Job was created to orchestrate the execution of the cloud pipeline.

The Job contains five dependent tasks:

```text
extract_kaggle.py
        ↓
preprocess.py
        ↓
feature_engineering.py
        ↓
train_model.py
        ↓
evaluate_model.py
```

Each task executes only after the previous task completes successfully.

Evidence of the Databricks Job configuration can be found in:

```text
screenshots/2_databricks_cloud_job.png
```

---

## Apache Airflow Integration

Apache Airflow acts as the orchestration layer.

Rather than executing Python modules directly, Airflow triggers the Azure Databricks Job using Databricks operators.

Workflow:

```text
Apache Airflow
        ↓
Azure Databricks Job
        ↓
PySpark Modules
        ↓
Azure Data Lake Storage
```

This architecture separates orchestration from execution:

* Airflow manages scheduling and orchestration.
* Databricks executes distributed workloads.
* ADLS stores datasets, intermediate outputs, models, and evaluation artifacts.

Evidence of successful Airflow-to-Databricks orchestration can be found in:

```text
screenshots/3_airflow_databricks_orchestration.png
```

---

# Module Descriptions

## extract_kaggle.py

Responsibilities:

* Connect to Kaggle using the Kaggle API.
* Download the Credit Card Fraud Detection dataset.
* Extract the dataset contents.
* Load raw data into the processing environment.

Output:

* Raw fraud detection dataset.

---

## preprocess.py

Responsibilities:

* Validate dataset structure.
* Remove duplicate transactions.
* Perform preprocessing operations.
* Create the Log_Amount feature.
* Validate data quality.

Output:

* Processed dataset ready for feature engineering.

---

## feature_engineering.py

Responsibilities:

* Assemble model features.
* Apply feature transformations.
* Apply RobustScaler.
* Generate model-ready feature vectors.

Output:

* Feature dataset ready for training.

---

## train_model.py

Responsibilities:

* Create train/test datasets.
* Handle class imbalance through weighting.
* Train Logistic Regression.
* Train Random Forest.
* Persist trained models.

Output:

* Trained machine learning models.
* Train/test datasets.

---

## evaluate_model.py

Responsibilities:

* Load trained models.
* Generate predictions.
* Compute evaluation metrics.
* Generate confusion matrices.
* Perform threshold analysis.
* Compare model performance.
* Persist evaluation artifacts.

Output:

* Model metrics.
* Threshold analysis results.
* Evaluation artifacts.

---

# Cloud Storage Architecture

The cloud pipeline uses Azure Data Lake Storage Gen2 (ADLS) as the central storage layer.

ADLS stores:

* Raw datasets
* Processed datasets
* Feature datasets
* Train/test datasets
* Trained models
* Evaluation metrics
* Threshold analysis results

This design allows each stage of the pipeline to consume outputs generated by previous stages while maintaining reproducibility and separation of concerns.

---

# Validation

Both implementations were successfully validated.

The repository therefore demonstrates:

* Local pipeline orchestration.
* Cloud pipeline orchestration.
* GitHub ↔ Azure Databricks integration.
* Apache Airflow ↔ Azure Databricks integration.
* Distributed processing with PySpark.
* Cloud storage using Azure Data Lake Storage Gen2.
* End-to-end reproducibility of the fraud detection workflow.
