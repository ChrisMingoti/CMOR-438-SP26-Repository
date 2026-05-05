# Feedforward Neural Network — Breast Cancer Classification

This notebook builds and trains a feedforward neural network from scratch using the `ml_pack` implementation to classify breast cancer tumors as malignant or benign. The network is implemented entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing and evaluation.

## Contents

* Overview of feedforward neural networks and key concepts
* Data loading and exploration
* Feature and target definition
* Train/test split and feature standardization
* Model training using `ml_pack.NeuralNetwork`
* Evaluation using accuracy, precision, recall, and F1 score
* Confusion matrix visualization
* ROC curve and AUC score
* Training loss curve

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