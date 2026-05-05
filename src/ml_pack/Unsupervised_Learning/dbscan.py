"""
dbscan.py

This module implements DBSCAN (Density-Based Spatial Clustering of Applications
with Noise) from scratch using NumPy. It is part of the ml_pack unsupervised
learning library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence


class DBSCAN:
    """
    A from-scratch implementation of the DBSCAN clustering algorithm.

    DBSCAN groups points that are closely packed together in high-density
    regions and marks points in low-density regions as outliers (noise).
    Unlike K-Means, it does not require the number of clusters to be
    specified in advance and can discover arbitrarily shaped clusters.

    The algorithm uses two parameters:
        - epsilon (eps): the radius of the neighborhood around each point
        - min_samples: the minimum number of points within eps for a point
          to be considered a core point

    Points are classified as:
        - Core points: have at least min_samples neighbors within eps
        - Border points: within eps of a core point but not core themselves
        - Noise points: not reachable from any core point, labeled as -1

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two points to be considered neighbors.
    min_samples : int, default=5
        The minimum number of points required within eps to form a core point.

    Attributes
    ----------
    labels : np.ndarray or None
        Cluster label for each point. Noise points are labeled -1.
        Available after calling fit().

    Examples
    --------
    >>> model = DBSCAN(eps=0.5, min_samples=5)
    >>> model.fit(X)
    >>> print(model.labels)
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")

        self.eps = eps
        self.min_samples = min_samples
        self.labels: Optional[np.ndarray] = None

    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Find all points within eps distance of a given point.

        Parameters
        ----------
        X : np.ndarray
            The full dataset.
        point_idx : int
            Index of the point to find neighbors for.

        Returns
        -------
        np.ndarray
            Indices of all neighboring points within eps.
        """
        # Compute Euclidean distances from the point to all other points
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        point_idx: int,
        neighbors: np.ndarray,
        cluster_id: int
    ) -> None:
        """
        Grow a cluster by adding all density-reachable points.

        Parameters
        ----------
        X : np.ndarray
            The full dataset.
        labels : np.ndarray
            Current label assignments.
        point_idx : int
            Index of the core point starting the cluster.
        neighbors : np.ndarray
            Initial neighbors of the core point.
        cluster_id : int
            The cluster ID to assign to this cluster.
        """
        labels[point_idx] = cluster_id

        # Use a queue to process all points reachable from this core point
        queue = list(neighbors)

        while queue:
            current = queue.pop(0)

            # If this point was previously labeled as noise, add it as a border point
            if labels[current] == -1:
                labels[current] = cluster_id

            # If this point hasn't been visited yet, assign it and check its neighbors
            elif labels[current] == -2:
                labels[current] = cluster_id
                current_neighbors = self._get_neighbors(X, current)

                # If it's a core point, add its neighbors to the queue
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)

    def fit(self, X: Union[np.ndarray, Sequence]) -> "DBSCAN":
        """
        Run DBSCAN on the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The dataset to cluster.

        Returns
        -------
        self : DBSCAN
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must not be empty.")

        n_samples = X.shape[0]

        # Initialize all labels as unvisited (-2)
        labels = np.full(n_samples, -2, dtype=int)

        cluster_id = 0

        for i in range(n_samples):
            # Skip already visited points
            if labels[i] != -2:
                continue

            neighbors = self._get_neighbors(X, i)

            # Not enough neighbors — mark as noise for now
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                # Core point — start a new cluster
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1

        self.labels = labels
        return self

    def fit_predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Run DBSCAN and return cluster labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The dataset to cluster.

        Returns
        -------
        np.ndarray
            Cluster label for each point. Noise points are labeled -1.
        """
        self.fit(X)
        return self.labels