# Principal Component Analysis — Breast Cancer Dataset

This notebook applies the `ml_pack` implementation of PCA to reduce the dimensionality of the Breast Cancer Wisconsin dataset from 30 features to a lower-dimensional representation. The algorithm is built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing and evaluation.

## Contents

* Overview of PCA and how it works
* Data loading and exploration
* Feature standardization
* Explained variance analysis
* Reducing to 2 components for visualization
* Visualizing class separation in the reduced space
* Using PCA as a preprocessing step to improve classifier accuracy

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