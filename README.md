# Task 1 — Credit Scoring Model

**Objective:** Predict creditworthiness using financial-history-style features.

**Dataset:** `credit_scoring_synthetic.csv` contains generated, fictional rows. It is provided so the
project runs without collecting personal financial data. Target `creditworthy`: 1 = creditworthy in
this educational example; 0 = higher risk.

**Algorithms:** Logistic Regression and Random Forest.

**Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and feature importance.

Run from the repository root:
`python Task1_Credit_Scoring/credit_scoring.py`

**Limitation:** Synthetic labels encode a simplified rule and do not reflect real-world lending.
Do not use this for actual credit decisions.
