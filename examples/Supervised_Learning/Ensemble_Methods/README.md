# Ensemble Methods — Heart Disease Prediction

This notebook applies the `ml_pack` implementations of Random Forest and Gradient Boosting to predict heart disease from clinical measurements. Both algorithms are built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing and evaluation.

## Contents

* Overview of ensemble methods and how they differ from single decision trees
* Data loading and exploration
* Feature and target definition
* Train/test split and feature standardization
* Training and evaluating `ml_pack.RandomForest`
* Training and evaluating `ml_pack.GradientBoosting`
* Confusion matrix comparison between both models
* Feature importance analysis

## Data

The **Heart Disease dataset** contains 303 patient records with 13 clinical features such as age, chest pain type, and maximum heart rate. The binary target variable indicates whether the patient has heart disease.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn (preprocessing and evaluation only)
* Matplotlib
* Seaborn
* ml_pack (see main README for installation)

## Usage

Run the notebook from top to bottom. Make sure `heart_disease.csv` is in the same directory as the notebook before running.