# Data

## Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle:

* Dataset: Credit Card Fraud Detection
* Source: Kaggle
* URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Dataset Overview

The dataset contains credit card transactions made by European cardholders and is commonly used for fraud detection research and machine learning benchmarking.

Main characteristics:

* Total records: 284,807 transactions
* Total variables: 31
* Target variable: `Class`

  * `0` = Non-fraudulent transaction
  * `1` = Fraudulent transaction
* Fraud cases: 492
* Fraud rate: approximately 0.17%
* Severe class imbalance (~578:1 non-fraud to fraud ratio)

## Dataset Structure

The dataset includes:

* `Time`: Seconds elapsed since the first transaction recorded in the dataset.
* `Amount`: Transaction amount.
* `Class`: Target variable.
* `V1`–`V28`: Anonymized features generated through Principal Component Analysis (PCA).

The original banking variables are not publicly available. Instead, the dataset provides 28 PCA-transformed components (`V1`–`V28`) that preserve transactional behavior patterns while protecting sensitive information.

Because these variables are already latent representations obtained through PCA, the dataset contains a preliminary level of feature engineering performed by the dataset creators.

## Project-Specific Feature Engineering

Although the dataset already includes PCA-transformed features, additional preprocessing and feature engineering were implemented in this project, including:

* Duplicate transaction removal.

* Creation of a log-transformed transaction amount feature:

  `Log_Amount = log(Amount + 1)`

* Robust scaling of numerical features using `RobustScaler`.

* Reproducible preprocessing artifacts for model training and future inference workflows.

These transformations were incorporated into the production-oriented machine learning pipeline developed in this repository.

## Data Availability

The raw dataset is **not included** in this repository.

This decision was made to:

* Respect dataset distribution policies.
* Keep the repository lightweight.
* Ensure full reproducibility through automated ingestion.

## Automated Data Ingestion

The pipeline automatically downloads the dataset from Kaggle using the Kaggle API.

During execution, the ingestion module:

1. Connects to Kaggle.
2. Downloads the dataset.
3. Extracts the compressed files.
4. Stores the raw data in the project's ingestion layer.
5. Launches downstream preprocessing and modeling stages.

As a result, users can reproduce the complete workflow without manually uploading the dataset.
