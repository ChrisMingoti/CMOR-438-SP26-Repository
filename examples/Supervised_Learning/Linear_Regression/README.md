# Linear Regression — Auto MPG

This notebook applies the `ml_pack` implementation of Linear Regression to predict car fuel efficiency using the Auto MPG dataset. The model is built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing and evaluation.

## Contents

* Overview of linear regression and its assumptions
* Data loading and exploration
* Handling missing values
* Feature and target variable definition
* Train/test split and feature standardization
* Model training and evaluation using `ml_pack.LinearRegression`
* Visualization of actual vs. predicted values
* Feature coefficient analysis

## Data

The **Auto MPG dataset** contains information on cars from the 1970s and 1980s. Each row represents a car model with features describing engine and physical properties. The target variable is **mpg** (miles per gallon), a continuous measure of fuel efficiency.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn (preprocessing and evaluation only)
* Matplotlib
* Seaborn
* ml_pack (see main README for installation)

## Usage

Run the notebook from top to bottom. Make sure `auto-mpg.csv` is in the same directory as the notebook before running.