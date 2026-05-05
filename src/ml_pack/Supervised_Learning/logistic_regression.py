"""
logistic_regression.py

This module implements Logistic Regression from scratch using gradient descent.
Supports binary classification with sigmoid activation and binary cross-entropy loss.
Part of the ml_pack supervised learning library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence, List


class LogisticRegression:
    """
    A from-scratch implementation of Logistic Regression for binary classification.

    Logistic regression models the probability that an input belongs to the
    positive class by applying the sigmoid function to a linear combination
    of input features. Parameters are learned by minimizing binary
    cross-entropy loss using gradient descent.

    The predicted probability is:
        p = sigmoid(X @ w + b) = 1 / (1 + exp(-(X @ w + b)))

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates.
    epochs : int, default=1000
        Number of full passes through the training data.
    random_state : int or None, default=None
        Seed for reproducible weight initialization.

    Attributes
    ----------
    weights : np.ndarray or None
        Learned feature coefficients. Set after calling fit().
    bias : float or None
        Learned intercept term. Set after calling fit().
    loss_history : list of float
        Training loss recorded at each epoch.

    Examples
    --------
    >>> model = LogisticRegression(learning_rate=0.01, epochs=1000)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        random_state: Optional[int] = None
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.loss_history: List[float] = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation — maps input to (0, 1)."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "LogisticRegression":
        """
        Train the logistic regression model using gradient descent.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : LogisticRegression
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            # Compute and store loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Compute gradients
            errors = y_pred - y
            grad_weights = (1 / n_samples) * (X.T @ errors)
            grad_bias = (1 / n_samples) * np.sum(errors)

            # Update parameters
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

        return self

    def predict_proba(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Return predicted probabilities for the positive class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted probabilities between 0 and 1.
        """
        if self.weights is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: Union[np.ndarray, Sequence], threshold: float = 0.5) -> np.ndarray:
        """
        Return binary class predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        threshold : float, default=0.5
            Probability threshold for classifying as positive.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> float:
        """
        Compute classification accuracy on the provided data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True binary labels.

        Returns
        -------
        float
            Proportion of correctly classified samples.
        """
        if self.weights is None:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y == y_pred))