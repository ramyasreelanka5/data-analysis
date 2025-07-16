# Loan Approval Prediction

This repository contains two projects for predicting loan approval status using Machine Learning:

1. Loan Approval Prediction using Random Forest
2. Loan Approval Prediction using Decision Tree

Both projects use a dataset of 1000 loan applications with features like income, loan amount, credit history, and more.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Models](#models)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Decision Tree Classifier](#decision-tree-classifier)
- [Usage](#usage)

---

## Overview

The goal is to predict whether a loan application will be approved ('Y') or rejected ('N') based on applicant details using supervised learning.

Two models are implemented:

- **Random Forest Classifier**
- **Decision Tree Classifier**

---

## Dataset

- CSV file with 1000 rows × 12 columns.
- Example columns:
  - Gender
  - Married
  - Dependents
  - Education
  - Self_Employed
  - ApplicantIncome
  - CoapplicantIncome
  - LoanAmount
  - Loan_Amount_Term
  - Credit_History
  - Property_Area
  - Loan_Status

Sample:

| Gender | Married | Dependents | Education   | Self_Employed | ApplicantIncome | CoapplicantIncome | LoanAmount | Loan_Amount_Term | Credit_History | Property_Area | Loan_Status |
|--------|---------|------------|-------------|---------------|-----------------|--------------------|------------|------------------|-----------------|----------------|-------------|
| Male   | No      | 2          | Not Graduate| No            | 19354           | 2875               | 186        | 360              | 1               | Urban          | N           |
| Female | Yes     | 4          | Not Graduate| Yes           | 7142            | 3191               | 69         | 60               | 0               | Urban          | Y           |

---

## Features Used

### For Random Forest Model

- Dependents
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History

### For Decision Tree Model

- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term

---

## Models

### Random Forest Classifier

- Imported from `sklearn.ensemble`
- Trained on 6 numeric features
- Predicts 'Y' or 'N'

Sample usage:

```python
model = RandomForestClassifier()
model.fit(feat, target)

# Predict for new input
pred = model.predict([[Dependents, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History]])
print(pred)

if pred == ['Y']:
    print("We are glad to inform that your Loan will be approved")
else:
    print("We are sorry, your Loan will not be approved")

```

### Decision Tree Classifier

- Imported from `sklearn.tree`
- Trained on 4 numeric features
- Visualized using `plot_tree()`

**Sample usage:**

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train the model
model = DecisionTreeClassifier()
model.fit(feat, target)

# Visualize the tree
plot_tree(model)

# Predict for new input
pred = model.predict([[ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term]])
print(pred)

if pred == ['Y']:
    print("We are glad to inform that your Loan will be approved")
else:
    print("We are sorry, your Loan will not be approved")
```

