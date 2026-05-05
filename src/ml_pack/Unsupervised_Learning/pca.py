"""
pca.py

This module implements Principal Component Analysis (PCA) from scratch using NumPy.
It is part of the ml_pack unsupervised learning library developed for CMOR 438
at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence


class PCA:
    """
    A from-scratch implementation of Principal Component Analysis (PCA).

    PCA reduces the dimensionality of data by projecting it onto the directions
    of maximum variance (principal components). These components are computed
    as the eigenvectors of the data's covariance matrix, ordered by their
    corresponding eigenvalues in descending order.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep. If None, all components
        are kept.

    Attributes
    ----------
    components : np.ndarray or None
        Principal axes in feature space, shape (n_components, n_features).
    explained_variance : np.ndarray or None
        Variance explained by each principal component.
    explained_variance_ratio : np.ndarray or None
        Proportion of total variance explained by each component.
    mean : np.ndarray or None
        Per-feature mean computed from the training data.

    Examples
    --------
    >>> model = PCA(n_components=2)
    >>> model.fit(X)
    >>> X_reduced = model.transform(X)
    >>> print(model.explained_variance_ratio)
    """

    def __init__(self, n_components: Optional[int] = None):
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be at least 1.")

        self.n_components = n_components

        self.components: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, Sequence]) -> "PCA":
        """
        Fit PCA on the input data by computing principal components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCA
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] < 2:
            raise ValueError("X must have at least 2 samples.")

        n_samples, n_features = X.shape

        # Validate n_components
        if self.n_components is not None and self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"the number of features ({n_features})."
            )

        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by descending eigenvalue
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Keep only n_components
        n = self.n_components if self.n_components is not None else n_features
        self.components = eigenvectors[:, :n].T
        self.explained_variance = eigenvalues[:n]
        self.explained_variance_ratio_ = self.explained_variance / np.sum(eigenvalues)

        return self

    def transform(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Project data onto the principal components.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Transformed data in the reduced space.
        """
        if self.components is None:
            raise RuntimeError("Model must be fitted before calling transform().")

        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Fit PCA and transform the data in one step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
            Transformed data in the reduced space.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Transform data back to the original feature space.

        Parameters
        ----------
        X_reduced : array-like, shape (n_samples, n_components)
            Data in the reduced space.

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Reconstructed data in the original feature space.
        """
        if self.components is None:
            raise RuntimeError("Model must be fitted before calling inverse_transform().")

        X_reduced = np.asarray(X_reduced, dtype=float)
        return X_reduced @ self.components + self.mean