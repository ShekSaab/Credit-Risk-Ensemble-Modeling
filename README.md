# Credit-Risk-Ensemble-Modeling
Applying Random Forests and Gradient Boosting to predict credit card default risk using the UCI "Default of Credit Card Clients" dataset.

# Credit Card Default Prediction: Ensemble Machine Learning
**MSc Financial Engineering | University of Glasgow**

## Project Overview
This project focuses on building and comparing advanced ensemble machine learning algorithms to forecast credit-card default risk. Using a dataset of 30,000 observations from Taiwan, the goal is to generate accurate and economically interpretable predictions to aid in credit-risk decision-making.

## Methodology

### 1. Data Preprocessing & Pipeline
Constructed a robust `scikit-learn` Pipeline to automate:
* **Categorical Encoding:** One-hot encoding for SEX, EDUCATION, and MARRIAGE.
* **Categorical Consolidation:** Irregular codes collapsed into a unified "Other" category.
* **Feature Scaling:** Standardization of all continuous predictors.
* **Imbalance Handling:** Strategic use of `class_weight='balanced'` or random over-sampling to address the minority default class.

### 2. Model Development & Hyper-parameter Tuning
Implemented 5-fold stratified cross-validation using `GridSearchCV` to optimize:
* **Random Forest:** Tuning tree count, maximum depth, and leaf sample requirements.
* **Gradient Boosting:** Optimizing learning rates, boosting stages, and subsample fractions.

### 3. Performance & Economic Interpretation
* **Metrics:** Evaluated models using ROC-AUC, F1-score, and confusion matrices.
* **Interpretability:** Utilized **SHAP (SHapley Additive exPlanations)** values to identify the top 10 most influential predictors of default.
* **Economic Intuition:** Analyzed borrower risk profiles to determine if data patterns align with standard credit-risk theory.

## Technical Environment
* **Language:** Python
* **Core Libraries:** Scikit-learn, Pandas, NumPy, SHAP, Matplotlib/Seaborn.
