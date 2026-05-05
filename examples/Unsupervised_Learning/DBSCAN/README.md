# DBSCAN — Mall Customer Segmentation

This notebook applies DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to the Mall Customer dataset to identify natural groupings of customers based on their spending behavior.

## Contents

* Overview of DBSCAN and how it differs from K-Means
* Data loading and exploration
* Feature selection and standardization
* Applying DBSCAN and interpreting cluster labels
* Visualization of clusters and outliers
* Analysis of how epsilon affects clustering results

## Data

The **Mall Customer dataset** contains information on 200 customers including their age, gender, annual income, and spending score. We cluster on **Annual Income** and **Spending Score** to identify behavioral segments.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* Seaborn

## Usage

Run the notebook from top to bottom. Make sure `Mall_Customers.csv` is in the same directory as the notebook before running.