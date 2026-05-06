# CMOR 438 Data Science & Machine Learning SP26

### Name: Christian Mingoti
### NetID: cm200
---

This repository documents my work for CMOR 438 at Rice University. It includes a custom machine learning package called **`ml_pack`** built entirely from scratch using Python and NumPy, alongside Jupyter notebooks that walk through the theory, math, and application of each algorithm covered in the course.

---

## Repository Structure

```
.
├── src/
│   └── ml_pack/
│       ├── Supervised_Learning/
│       ├── Unsupervised_Learning/
│       └── processing/
├── examples/
│   ├── Supervised_Learning/
│   │   ├── Linear_Regression/
│   │   ├── Logistic_Regression/
│   │   ├── Neural_Networks/
│   │   ├── KNN/
│   │   ├── Decision_Trees/
│   │   ├── Ensemble_Methods/
│   │   └── Naive_Bayes/
│   └── Unsupervised_Learning/
│       ├── DBSCAN/
│       ├── KMeans/
│       ├── PCA/
│       └── tSNE/
├── tests/unit
├── pyproject.toml
└── README.md
```

* **`src/ml_pack/`** — The core package. Each algorithm is written from scratch with a focus on understanding the underlying math rather than wrapping existing libraries.

* **`examples/`** — One notebook per algorithm covering data exploration, implementation walkthroughs, and model evaluation using ml_pack.

* **`tests/unit/`** — Unit tests for every implemented method using pytest.

* **`pyproject.toml`** — Project configuration and package installation settings.

---

## Algorithms Implemented

### Supervised Learning

* Linear Regression (OLS, Ridge, Gradient Descent)
* Logistic Regression
* k-Nearest Neighbors (kNN)
* Decision Trees (CART with Gini impurity)
* Ensemble Methods (Random Forest, Gradient Boosting)
* Neural Networks (Feedforward MLP with backpropagation)
* Naïve Bayes (Gaussian)

### Unsupervised Learning

* K-Means Clustering
* DBSCAN
* Principal Component Analysis (PCA)
* t-SNE

### Data Processing Utilities

* StandardScaler
* train_test_split
* Regression metrics (MSE, RMSE, MAE, R²)
* Classification metrics (accuracy, confusion matrix)

---

## Installation

Clone the repo and install the package locally:

pip install -e .

---

## Running Tests

python -m pytest

---