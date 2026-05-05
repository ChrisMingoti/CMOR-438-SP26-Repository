# Logistic Regression — Breast Cancer Classification

This notebook applies the `ml_pack` implementation of Logistic Regression to classify breast cancer tumors as malignant or benign. The algorithm is built entirely from scratch in `ml_pack` using NumPy with gradient descent optimization, with sklearn used only for preprocessing and evaluation.

## Contents

* Overview of logistic regression and the sigmoid function
* Data loading and exploration
* Feature and target definition
* Train/test split and feature standardization
* Model training using `ml_pack.LogisticRegression`
* Evaluation using precision, recall, and F1 score
* Confusion matrix and ROC curve visualization
* Feature coefficient analysis

## Data

The **Breast Cancer Wisconsin dataset** contains 569 samples with 30 numeric features derived from cell nucleus measurements. The binary target variable indicates whether a tumor is malignant (0) or benign (1).

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn (preprocessing and evaluation only)
* Matplotlib
* Seaborn
* ml_pack (see main README for installation)

## Usage

Run the notebook from top to bottom. Make sure `breast_cancer.csv` is in the same directory as the notebook before running.