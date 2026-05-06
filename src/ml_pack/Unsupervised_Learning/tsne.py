"""
tsne.py

This module implements t-SNE (t-distributed Stochastic Neighbor Embedding)
from scratch using NumPy. It is part of the ml_pack unsupervised learning
library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence


class TSNE:
    """
    A from-scratch implementation of t-SNE dimensionality reduction.

    t-SNE reduces high-dimensional data to a lower-dimensional space
    (typically 2D or 3D) for visualization. It preserves local structure
    by keeping similar points close together in the reduced space.

    The algorithm works in two steps:
        1. Compute pairwise affinities in the high-dimensional space using
           a Gaussian kernel with perplexity-based bandwidth selection
        2. Minimize the KL divergence between high-dimensional and
           low-dimensional affinities using gradient descent, where the
           low-dimensional distribution uses a Student t-distribution

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the output embedding.
    perplexity : float, default=30.0
        Controls the effective number of neighbors considered. Typical
        values are between 5 and 50.
    learning_rate : float, default=200.0
        Step size for gradient descent.
    n_iter : int, default=1000
        Number of gradient descent iterations.
    random_state : int or None, default=None
        Seed for reproducible initialization.

    Attributes
    ----------
    embedding : np.ndarray or None
        Low-dimensional embedding of the input data. Set after calling fit().

    Examples
    --------
    >>> model = TSNE(n_components=2, perplexity=30, random_state=42)
    >>> X_reduced = model.fit_transform(X)
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: Optional[int] = None
    ):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        if perplexity <= 0:
            raise ValueError("perplexity must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")

        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

        self.embedding: Optional[np.ndarray] = None

    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances between all pairs of points."""
        sum_sq = np.sum(X ** 2, axis=1)
        distances = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * (X @ X.T)
        return np.maximum(distances, 0)

    def _compute_joint_probabilities(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute joint probabilities P in the high-dimensional space.

        Uses binary search to find the bandwidth (sigma) for each point
        such that the resulting distribution has the desired perplexity.
        """
        n = distances.shape[0]
        target_entropy = np.log(self.perplexity)
        P = np.zeros((n, n))

        for i in range(n):
            # Binary search for sigma
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0

            dist_i = distances[i].copy()
            dist_i[i] = np.inf  # exclude self

            for _ in range(50):
                exp_d = np.exp(-dist_i * beta)
                exp_d[i] = 0
                sum_exp = np.sum(exp_d) + 1e-10

                # Compute entropy
                p_i = exp_d / sum_exp
                entropy = -np.sum(p_i[p_i > 0] * np.log(p_i[p_i > 0] + 1e-10))

                diff = entropy - target_entropy
                if abs(diff) < 1e-5:
                    break

                if diff > 0:
                    beta_min = beta
                    beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
                else:
                    beta_max = beta
                    beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

            P[i] = p_i

        # Symmetrize and normalize
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        return P

    def _compute_low_dim_affinities(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute affinities Q in the low-dimensional space using Student t-distribution.
        """
        distances = self._compute_pairwise_distances(Y)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q = Q / (np.sum(Q) + 1e-10)
        return np.maximum(Q, 1e-12)

    def fit_transform(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Fit t-SNE and return the low-dimensional embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            High-dimensional input data.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] < 2:
            raise ValueError("X must have at least 2 samples.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n = X.shape[0]

        # Compute high-dimensional affinities
        distances = self._compute_pairwise_distances(X)
        P = self._compute_joint_probabilities(distances)

        # Initialize low-dimensional embedding randomly
        Y = np.random.randn(n, self.n_components) * 0.01
        Y_prev = Y.copy()
        momentum = 0.5

        for iteration in range(self.n_iter):
            # Switch to higher momentum after 250 iterations
            if iteration == 250:
                momentum = 0.8

            # Compute low-dimensional affinities
            Q = self._compute_low_dim_affinities(Y)

            # Compute gradient of KL divergence
            PQ_diff = (P - Q)[:, :, np.newaxis]
            diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
            dist_factor = (1 / (1 + self._compute_pairwise_distances(Y)))[:, :, np.newaxis]
            grad = 4 * np.sum(PQ_diff * diff * dist_factor, axis=1)

            # Update embedding with momentum
            Y_new = Y - self.learning_rate * grad + momentum * (Y - Y_prev)
            Y_prev = Y.copy()
            Y = Y_new

            # Re-center embedding
            Y -= Y.mean(axis=0)

        self.embedding = Y
        return Y

    def fit(self, X: Union[np.ndarray, Sequence]) -> "TSNE":
        """
        Fit t-SNE on the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            High-dimensional input data.

        Returns
        -------
        self : TSNE
        """
        self.fit_transform(X)
        return self