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
│       ├── supervised_learning/
│       ├── unsupervised_learning/
│       └── processing/
├── examples/
│   ├── supervised_learning/
│   └── unsupervised_learning/
├── tests/unit
├── pyproject.toml
├── requirements.txt
└── README.md
```

* **`src/ml_pack/`** — The core package. Each algorithm is written from scratch with a focus on understanding the underlying math rather than wrapping existing libraries.

* **`notebooks/`** — One notebook per algorithm, covering data exploration, implementation walkthroughs, and model evaluation.

* **`tests/unit/`** — Unit tests for every implemented method using pytest.

* **`examples/`** — Notebooks showing ml_pack used end-to-end on real datasets.

* **`requirements.txt`** — All packages needed to run this repo.

---

## Algorithms Implemented

### Supervised Learning

* Linear Regression
* Logistic Regression
* k-Nearest Neighbors (kNN)
* Decision Trees
* Ensemble Methods
* Neural Networks

### Unsupervised Learning

* K-Means Clustering
* DBSCAN
* Principal Component Analysis (PCA)

---

## Installation

Clone the repo and install the package locally:

pip install -e .

---

## Running Tests

python -m pytest

---