# K-Means Clustering — Mall Customer Segmentation

This notebook applies K-Means clustering to segment mall customers based on their annual income and spending score.

## Contents

* Overview of K-Means and how it differs from DBSCAN
* Data loading and exploration
* Feature selection and standardization
* Elbow method to determine optimal number of clusters
* Applying K-Means with k=5
* Visualization of clusters and centroids
* Interpretation of customer segments

## Data

The **Mall Customer dataset** contains information on 200 customers including their age, gender, annual income, and spending score. We cluster on **Annual Income** and **Spending Score** to identify distinct customer personas.

## Requirements

* Python 3.x
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* Seaborn

## Usage

Run the notebook from top to bottom. Make sure `Mall_Customers.csv` is in the same directory as the notebook before running.