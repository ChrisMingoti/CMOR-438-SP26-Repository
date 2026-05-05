"""
knn.py

This module implements the K-Nearest Neighbors algorithm from scratch using NumPy.
Supports both classification and regression tasks. Part of the ml_pack supervised
learning library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence
from collections import Counter


class KNN:
    """
    A from-scratch implementation of the K-Nearest Neighbors algorithm.

    KNN classifies a new data point by finding the k closest points in the
    training set and assigning the majority class among those neighbors.
    It makes no assumptions about the underlying data distribution, making
    it a flexible non-parametric method.

    Distance is computed using Euclidean distance by default. Feature scaling
    is strongly recommended before using this classifier.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors to consider.

    Attributes
    ----------
    X_train : np.ndarray or None
        Training feature matrix. Set after calling fit().
    y_train : np.ndarray or None
        Training labels. Set after calling fit().

    Examples
    --------
    >>> model = KNN(k=5)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "KNN":
        """
        Store the training data — KNN is a lazy learner so no computation happens here.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Training labels.

        Returns
        -------
        self : KNN
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] < self.k:
            raise ValueError(
                f"Number of training samples ({X.shape[0]}) must be >= k ({self.k})."
            )

        self.X_train = X
        self.y_train = y
        return self

    def _get_neighbors(self, x: np.ndarray) -> np.ndarray:
        """
        Find the k nearest neighbors for a single input point.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            A single input sample.

        Returns
        -------
        np.ndarray
            Labels of the k nearest neighbors.
        """
        # Compute Euclidean distances to all training points
        distances = np.linalg.norm(self.X_train - x, axis=1)

        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        return self.y_train[k_indices]

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix to predict on.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if self.X_train is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)

        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            # Majority vote among neighbors
            most_common = Counter(neighbors).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def score(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> float:
        """
        Compute classification accuracy on the provided data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True labels.

        Returns
        -------
        float
            Proportion of correctly classified samples.
        """
        if self.X_train is None:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y == y_pred))