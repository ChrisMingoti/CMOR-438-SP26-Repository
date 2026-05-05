# Feedforward Neural Network — Breast Cancer Classification

This notebook builds and trains a feedforward neural network from scratch using NumPy to classify breast cancer tumors as malignant or benign.

## Contents

* Overview of feedforward neural networks and key concepts
* Data loading and exploration
* Feature and target definition
* Train/test split and feature standardization
* Implementation of forward pass, loss function, and backpropagation
* Model training with batch gradient descent
* Evaluation using accuracy, precision, recall, and F1 score
* Confusion matrix visualization
* Training loss curve

## Data

The **Breast Cancer Wisconsin dataset** contains 569 samples with 30 numeric features derived from cell nucleus measurements. The binary target variable indicates whether a tumor is malignant (0) or benign (1).

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* Seaborn

## Usage

Run the notebook from top to bottom. Make sure `breast_cancer.csv` is in the same directory as the notebook before running.