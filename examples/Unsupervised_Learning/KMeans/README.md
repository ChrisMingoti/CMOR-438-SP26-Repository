# K-Means Clustering — Mall Customer Segmentation

This notebook applies the `ml_pack` implementation of K-Means to segment mall customers based on their annual income and spending score. The algorithm is built entirely from scratch in `ml_pack` using NumPy, with sklearn used only for preprocessing.

## Contents

* Overview of K-Means and how it differs from DBSCAN
* Data loading and exploration
* Feature selection and standardization
* Elbow method to determine optimal number of clusters
* Applying K-Means with k=5 using `ml_pack.KMeans`
* Visualization of clusters and centroids
* Interpretation of customer segments

## Data

The **Mall Customer dataset** contains information on 200 customers including their age, gender, annual income, and spending score. We cluster on **Annual Income** and **Spending Score** to identify distinct customer personas.

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