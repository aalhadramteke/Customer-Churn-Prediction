# Customer Churn Prediction

A reproducible machine learning analysis of customer churn in a telecommunications dataset. The project combines exploratory data analysis, feature preparation, visualization, and comparative classification modeling to identify customers who are likely to leave a service provider.

## Project Overview

Customer retention is a central business problem for subscription-based services. This project investigates customer demographics, account details, contract types, payment methods, service usage, and billing behavior to understand churn patterns and build baseline predictive models.

The analysis is implemented in [`main.ipynb`](main.ipynb), which walks through the complete workflow:

- Inspecting and cleaning the customer data
- Exploring churn distributions and customer behavior
- Encoding categorical features and standardizing numeric features
- Splitting data into stratified training and test sets
- Training and comparing five classification models
- Evaluating accuracy, precision, recall, F1 score, and confusion matrices

## Key Findings

- Approximately 26.6% of customers in the analyzed data churned.
- Month-to-month customers showed substantially higher churn than customers on one- or two-year contracts.
- Electronic check was associated with comparatively high churn.
- Customers with higher monthly charges were more likely to churn.
- Logistic Regression achieved the strongest reported accuracy: **0.812**.
- Decision Tree and SVC had the lowest reported accuracies: **0.732** and **0.734**, respectively.

Accuracy is not sufficient on its own for churn decisions. Recall and F1 score should also be considered because missing a customer who is likely to churn may be more costly than contacting a low-risk customer.

## Models Evaluated

| Model | Configuration / Notes |
| --- | --- |
| K-Nearest Neighbors | `n_neighbors=11` |
| Support Vector Classifier | `random_state=1` |
| Random Forest | 500 estimators, out-of-bag scoring, square-root feature sampling |
| Logistic Regression | Standardized pipeline, `max_iter=1000` |
| Decision Tree | Default classifier configuration |

## Dataset

The notebook expects the Telco Customer Churn CSV dataset, commonly distributed as `Telecom Customer Data Churn.csv`. The dataset contains customer account, service, demographic, and billing attributes, with `Churn` as the target variable.

The following fields are used during analysis, among others:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `PaymentMethod`
- `InternetService`
- `Churn`

The original `customerID` field is removed before modeling. The dataset is not included in this repository; obtain it from an authorized source and place it in the project directory or update the path in the notebook.

## Getting Started

### Requirements

- Python 3.9 or newer
- Jupyter Notebook or VS Code with the Jupyter extension

### Installation

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy seaborn matplotlib plotly scikit-learn
```

### Run the Analysis

1. Place `Telecom Customer Data Churn.csv` in the project directory.
2. Open [`main.ipynb`](main.ipynb) in Jupyter or VS Code.
3. Update the `pd.read_csv(...)` path in the data-loading cell if necessary.
4. Run all cells from top to bottom.

The notebook currently contains a machine-specific Windows path in the data-loading cell. For a portable clone, replace it with a relative path such as:

```python
df = pd.read_csv("Telecom Customer Data Churn.csv")
```

## Methodology

1. Load the customer churn dataset.
2. Remove the identifier column and coerce `TotalCharges` to numeric values.
3. Remove records with zero tenure and inspect missing values.
4. Explore churn by contract, payment method, gender, and charge distributions.
5. Encode categorical variables with one-hot encoding and scale numeric variables where appropriate.
6. Create a stratified 70/30 train-test split using `random_state=40`.
7. Train five baseline classifiers.
8. Compare predictions using accuracy, precision, recall, and F1 score.

## Repository Structure

```text
.
├── main.ipynb    # End-to-end analysis and model evaluation
└── README.md     # Project documentation
```

## Limitations and Next Steps

This is an analytical baseline rather than a production scoring service. The current notebook does not include hyperparameter search, cross-validation, probability calibration, model serialization, or deployment code. Future improvements could include:

- Cross-validated hyperparameter tuning
- ROC-AUC and precision-recall evaluation
- Explicit treatment of class imbalance
- Feature importance and model explainability
- A reusable preprocessing and inference pipeline
- Monitoring for data drift after deployment

