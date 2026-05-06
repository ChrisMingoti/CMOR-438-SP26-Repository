# t-SNE — Breast Cancer Dataset Visualization

This notebook applies the `ml_pack` implementation of t-SNE to visualize the Breast Cancer Wisconsin dataset in 2D and compares it to PCA. The algorithm is built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for PCA comparison and preprocessing.

## Contents

* Overview of t-SNE and how it differs from PCA
* Data loading and preprocessing
* Applying t-SNE using `ml_pack.TSNE`
* Visualizing the t-SNE embedding
* Side by side comparison of t-SNE and PCA embeddings

## Data

The **Breast Cancer Wisconsin dataset** contains 569 samples with 30 numeric features derived from cell nucleus measurements. The binary target variable indicates whether a tumor is malignant (0) or benign (1).

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn (PCA comparison and preprocessing only)
* Matplotlib
* Seaborn
* ml_pack (see main README for installation)

## Usage

Run the notebook from top to bottom. Make sure `breast_cancer.csv` is in the same directory as the notebook before running.