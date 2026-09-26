# Logistic Regression

This folder contains the practical implementation of Logistic Regression covered in the accompanying Medium article.

## Contents

### `Logistic_Regression.ipynb`
End-to-end implementation covering:

- Data loading
- Exploratory Data Analysis
- Train-test split
- Data preprocessing
- Logistic Regression
- Probability prediction
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Threshold tuning
- Coefficient interpretation

### `logistic_regression_scratch.py`

Logistic Regression implemented from scratch using NumPy.

Includes:

- Sigmoid function
- Binary Cross-Entropy
- Gradient calculation
- Gradient Descent
- Probability prediction
- Class prediction
- Numerically stable calculations

### `logistic_regression_pipeline.py`

Production-style implementation using Scikit-Learn.

Includes:

- Missing-value imputation
- Feature scaling
- One-hot encoding
- ColumnTransformer
- Pipeline
- Logistic Regression

## Mathematical Foundation

The model uses:

z = β₀ + Xβ

p = 1 / (1 + e⁻ᶻ)

Binary Cross-Entropy:

J = −(1/n) Σ[y log(p) + (1−y) log(1−p)]

Gradient:

∇J = Xᵀ(p − y) / n

Parameter update:

β ← β − α∇J

## Related Article

The complete mathematical explanation and intuition are covered in the accompanying Medium article.

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter