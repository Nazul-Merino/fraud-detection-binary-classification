                                                   ##   Exploratory Analysis Notebooks
                                          
This directory contains the exploratory analysis, data quality assessment, feature engineering experiments, model development, and evaluation workflows that served as the analytical foundation of the fraud detection project.

The notebooks were developed to understand the structure and behavior of the Credit Card Fraud Detection dataset, identify potential data quality issues, evaluate preprocessing alternatives, investigate fraud-related patterns, and compare candidate machine learning models before transitioning to a production-oriented architecture.

The exploratory analysis revealed several important characteristics of the dataset. The target variable exhibited extreme class imbalance, with fraudulent transactions representing approximately 0.17% of all observations, corresponding to an imbalance ratio close to 578:1. This finding established the need for imbalance-aware evaluation strategies and motivated the use of metrics such as Precision, Recall, F1-score, ROC-AUC, and PR-AUC instead of relying on accuracy alone.

Data quality assessment showed that the dataset contained no missing values or infinite values and was structurally well suited for machine learning workflows. However, duplicate records were identified and later incorporated into the preprocessing strategy. Additional exploratory analyses revealed substantial skewness and outlier behavior across several variables, leading to a conservative preprocessing philosophy focused on preserving potentially informative fraud-related signals rather than aggressively removing extreme observations.
Feature engineering experiments demonstrated that RobustScaler provided greater resistance to outlier effects than alternative scaling approaches, while logarithmic transformation of transaction amounts improved distribution interpretability and downstream modeling stability. The analysis also confirmed that the PCA-transformed variables already contained substantial predictive information, motivating a deliberately conservative feature engineering strategy.

Modeling experiments highlighted the importance of threshold selection and imbalance-aware evaluation. Logistic Regression and Random Forest exhibited substantially different operational behaviors, revealing the tradeoff between fraud detection sensitivity and false positive management. The findings also showed that fraud patterns were highly nonlinear and interaction-driven, supporting the later inclusion of ensemble-based approaches.

Feature importance analyses consistently identified variables such as V14, V12, V10, V17, and V4 as the strongest fraud indicators across multiple analytical methods. These findings contributed directly to model interpretation, evaluation, and final architecture decisions.
The insights generated throughout these notebooks were subsequently used to define the production-oriented preprocessing strategy, feature engineering workflow, model evaluation framework, cloud architecture design, and orchestration strategy implemented in the later phases of the project.

Readers interested in the complete technical findings may refer to:

•	docs/phase_1_exploratory_analysis_summary.pdf

•	docs/phase_2_production_architecture_summary.pdf

