# K-Nearest Neighbors — Iris Flower Classification

This notebook applies the `ml_pack` implementation of K-Nearest Neighbors to classify iris flowers into three species based on sepal and petal measurements. The KNN classifier is built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing and evaluation.

## Contents

* Overview of kNN and how it works
* Data loading and exploration
* Feature and target definition
* Train/test split and feature standardization
* Finding the optimal k using accuracy plots
* Model training and evaluation using `ml_pack.KNN`
* Confusion matrix visualization

## Data

The **Iris dataset** contains 150 samples with 4 numeric features describing sepal and petal dimensions. The target variable is the flower species — setosa, versicolor, or virginica — with 50 samples per class.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn (preprocessing and evaluation only)
* Matplotlib
* Seaborn
* ml_pack (see main README for installation)

## Usage

Run the notebook from top to bottom. Make sure `iris.csv` is in the same directory as the notebook before running.