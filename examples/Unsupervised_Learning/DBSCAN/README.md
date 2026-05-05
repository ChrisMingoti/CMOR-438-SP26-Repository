# DBSCAN — Mall Customer Segmentation

This notebook applies the `ml_pack` implementation of DBSCAN to identify natural groupings and outliers in the Mall Customer dataset. The algorithm is built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing.

## Contents

* Overview of DBSCAN and how it differs from K-Means
* Data loading and exploration
* Feature selection and standardization
* Applying DBSCAN and interpreting cluster labels using `ml_pack.DBSCAN`
* Visualization of clusters and outliers
* Analysis of how epsilon affects clustering results
* Outlier profile analysis

## Data

The **Mall Customer dataset** contains information on 200 customers including their age, gender, annual income, and spending score. We cluster on **Annual Income** and **Spending Score** to identify behavioral segments.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn (preprocessing only)
* Matplotlib
* Seaborn
* ml_pack (see main README for installation)

## Usage

Run the notebook from top to bottom. Make sure `Mall_Customers.csv` is in the same directory as the notebook before running.