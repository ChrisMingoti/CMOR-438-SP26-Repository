"""
naive_bayes.py

This module implements Gaussian Naive Bayes from scratch using NumPy.
It is part of the ml_pack supervised learning library developed for
CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence


class GaussianNaiveBayes:
    """
    A from-scratch implementation of Gaussian Naive Bayes classification.

    Gaussian Naive Bayes applies Bayes' theorem with the assumption that
    features are conditionally independent given the class label, and that
    each feature follows a Gaussian (normal) distribution within each class.

    For each class, the algorithm estimates:
        - Prior probability: P(class) = proportion of training samples in that class
        - Likelihood: P(feature | class) modeled as a Gaussian distribution

    Predictions are made by computing the posterior probability for each class
    and returning the class with the highest posterior.

    Attributes
    ----------
    classes : np.ndarray or None
        Unique class labels. Set after calling fit().
    priors : np.ndarray or None
        Prior probability of each class.
    means : np.ndarray or None
        Per-class mean of each feature. Shape (n_classes, n_features).
    variances : np.ndarray or None
        Per-class variance of each feature. Shape (n_classes, n_features).

    Examples
    --------
    >>> model = GaussianNaiveBayes()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(self):
        self.classes: Optional[np.ndarray] = None
        self.priors: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.variances: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "GaussianNaiveBayes":
        """
        Train the model by computing class priors, means, and variances.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Training labels.

        Returns
        -------
        self : GaussianNaiveBayes
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        n_samples = X.shape[0]

        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes):
            # Select samples belonging to this class
            X_cls = X[y == cls]

            # Compute prior, mean and variance for each feature
            self.priors[i] = X_cls.shape[0] / n_samples
            self.means[i] = np.mean(X_cls, axis=0)
            self.variances[i] = np.var(X_cls, axis=0) + 1e-9  # small value to avoid division by zero

        return self

    def _log_likelihood(self, X: np.ndarray, class_idx: int) -> np.ndarray:
        """
        Compute the log likelihood of each sample under a given class.

        Uses the Gaussian probability density function:
            log P(x | class) = sum over features of log N(x; mean, variance)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        class_idx : int
            Index of the class to compute likelihood for.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Log likelihood for each sample.
        """
        mean = self.means[class_idx]
        var = self.variances[class_idx]

        # Log of Gaussian PDF
        log_prob = -0.5 * np.sum(np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)
        return log_prob

    def predict_proba(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Return posterior probabilities for each class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Posterior probabilities for each class.
        """
        if self.classes is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        X = np.asarray(X, dtype=float)

        log_posteriors = np.array([
            np.log(self.priors[i]) + self._log_likelihood(X, i)
            for i in range(len(self.classes))
        ]).T

        # Convert log posteriors to probabilities
        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
        probs = np.exp(log_posteriors)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if self.classes is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

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
        if self.classes is None:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y)
        return float(np.mean(y == self.predict(X)))