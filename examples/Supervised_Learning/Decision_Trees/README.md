# Decision Trees — Wine Quality Classification

This notebook applies the `ml_pack` implementation of a Decision Tree classifier to predict whether a red wine is high or low quality based on its chemical properties. The algorithm is built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing and evaluation.

## Contents

* Overview of decision trees and how they work
* Data loading and exploration
* Binary target creation (high vs low quality)
* Train/test split and feature standardization
* Model training using `ml_pack.DecisionTree`
* Evaluation using precision, recall, and F1 score
* Confusion matrix visualization
* Feature importance analysis

## Data

The **Wine Quality dataset** contains 1599 red wine samples with 11 chemical features such as alcohol content, acidity, and sulphates. The target variable is binary — wines rated 7 or above are considered high quality.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn (preprocessing and evaluation only)
* Matplotlib
* Seaborn
* ml_pack (see main README for installation)

## Usage

Run the notebook from top to bottom. Make sure `wine_quality.csv` is in the same directory as the notebook before running.