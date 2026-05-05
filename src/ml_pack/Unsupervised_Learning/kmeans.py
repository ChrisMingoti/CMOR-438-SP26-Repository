"""
kmeans.py

This module implements the K-Means clustering algorithm from scratch using NumPy.
It is part of the ml_pack unsupervised learning library developed for CMOR 438
at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence


class KMeans:
    """
    A from-scratch implementation of the K-Means clustering algorithm.

    K-Means partitions data into k clusters by iteratively assigning each point
    to the nearest centroid and updating centroids as the mean of assigned points.
    This process repeats until centroids stop moving or the maximum number of
    iterations is reached.

    Parameters
    ----------
    n_clusters : int, default=5
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations before stopping.
    tol : float, default=1e-4
        Convergence threshold — stops early if centroids move less than this.
    random_state : int or None, default=None
        Seed for reproducible centroid initialization.

    Attributes
    ----------
    centroids : np.ndarray or None
        Final centroid positions. Shape (n_clusters, n_features).
    labels : np.ndarray or None
        Cluster label for each input sample. Set after calling fit().
    inertia : float or None
        Sum of squared distances from each point to its assigned centroid.
    n_iter : int
        Number of iterations run before convergence.

    Examples
    --------
    >>> model = KMeans(n_clusters=5, random_state=42)
    >>> model.fit(X)
    >>> print(model.labels)
    >>> print(model.inertia)
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.inertia: Optional[float] = None
        self.n_iter: int = 0

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Randomly select k data points as initial centroids.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray of shape (n_clusters, n_features)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to the nearest centroid using Euclidean distance.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        centroids : np.ndarray, shape (n_clusters, n_features)
            Current centroid positions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Cluster label for each point.
        """
        # Compute distance from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Recompute centroids as the mean of all points assigned to each cluster.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        labels : np.ndarray
            Current cluster assignments.

        Returns
        -------
        np.ndarray of shape (n_clusters, n_features)
        """
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
            for k in range(self.n_clusters)
        ])
        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """
        Compute the sum of squared distances from each point to its centroid.
        """
        return float(sum(
            np.sum((X[labels == k] - centroids[k]) ** 2)
            for k in range(self.n_clusters)
            if np.any(labels == k)
        ))

    def fit(self, X: Union[np.ndarray, Sequence]) -> "KMeans":
        """
        Run K-Means on the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The dataset to cluster.

        Returns
        -------
        self : KMeans
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"Number of samples ({X.shape[0]}) must be >= n_clusters ({self.n_clusters})."
            )

        centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            labels = self._assign_labels(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            # Check for convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                self.n_iter = i + 1
                break
        else:
            self.n_iter = self.max_iter

        self.centroids = centroids
        self.labels = labels
        self.inertia = self._compute_inertia(X, labels, centroids)

        return self

    def fit_predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Run K-Means and return cluster labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The dataset to cluster.

        Returns
        -------
        np.ndarray
            Cluster label for each point.
        """
        self.fit(X)
        return self.labels

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Assign new data points to the nearest centroid.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to assign.

        Returns
        -------
        np.ndarray
            Cluster label for each point.
        """
        if self.centroids is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.centroids)