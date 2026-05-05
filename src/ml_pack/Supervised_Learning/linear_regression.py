"""
linear_regression.py

This module implements Linear Regression from scratch using three approaches:
Ordinary Least Squares (OLS), Ridge Regression (L2 regularization), and 
Batch Gradient Descent. It is part of the ml_pack supervised learning library 
developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence, Literal, List


class LinearRegression:
    """
    A from-scratch implementation of Linear Regression supporting three solvers.

    The model learns a linear mapping from input features to a continuous target
    by minimizing the residual sum of squares. Three solvers are available:

        'ols'   - Closed-form solution using the normal equation
        'ridge' - Closed-form with L2 penalty to control model complexity
        'gd'    - Iterative optimization via Batch Gradient Descent

    Parameters
    ----------
    method : {'ols', 'ridge', 'gd'}, default='ols'
        Which solver to use during training.
    alpha : float, default=0.0
        Regularization strength for Ridge. Larger values shrink weights more.
        Only used when method='ridge'.
    learning_rate : float, default=0.01
        Controls the step size in Gradient Descent.
        Only used when method='gd'.
    epochs : int, default=1000
        Maximum number of Gradient Descent iterations.
        Only used when method='gd'.
    random_state : int or None, default=None
        Random seed for reproducible weight initialization in 'gd' mode.

    Attributes
    ----------
    weights : np.ndarray or None
        Coefficients for each input feature. Available after fitting.
    bias : float or None
        Intercept term. Available after fitting.
    cost_history : list of float
        Training loss recorded at each epoch. Only populated when method='gd'.

    Examples
    --------
    >>> model = LinearRegression(method='ols')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        method: Literal['ols', 'ridge', 'gd'] = 'ols',
        alpha: float = 0.0,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        random_state: Optional[int] = None
    ):
        if method not in ('ols', 'ridge', 'gd'):
            raise ValueError(f"method must be 'ols', 'ridge', or 'gd'. Got '{method}'.")
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.method = method
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.cost_history: List[float] = []

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Convert inputs to float arrays and validate their shapes."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got {X.shape[0]} and {y.shape[0]}."
            )
        return X, y

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using the OLS normal equation.

        Appends a bias column to X, then solves:
            w = pinv(X^T X) X^T y
        """
        # Append a column of ones to handle the bias term
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])

        # Solve the normal equation using pseudo-inverse for numerical stability
        XTX = X_b.T @ X_b
        w = np.linalg.pinv(XTX) @ X_b.T @ y

        self.weights = w[:-1]
        self.bias = float(w[-1])

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using Ridge regression (OLS + L2 regularization).

        Solves:
            w = pinv(X^T X + alpha * I) X^T y

        The bias term is not regularized.
        """
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])

        XTX = X_b.T @ X_b

        # Build regularization matrix — exclude the bias term from penalization
        reg_matrix = self.alpha * np.eye(X_b.shape[1])
        reg_matrix[-1, -1] = 0.0

        w = np.linalg.pinv(XTX + reg_matrix) @ X_b.T @ y

        self.weights = w[:-1]
        self.bias = float(w[-1])

    def _fit_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit using Batch Gradient Descent.

        At each epoch, computes gradients over the full dataset and
        updates weights and bias using the learning rate.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize weights and bias with small random values
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = float(np.random.randn() * 0.01)
        self.cost_history = []

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            errors = y_pred - y

            # Compute gradients
            grad_weights = (1 / n_samples) * (X.T @ errors)
            grad_bias = (1 / n_samples) * np.sum(errors)

            # Update parameters
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

            # Track MSE cost (scaled by 1/2 for cleaner gradient math)
            cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "LinearRegression":
        """
        Train the model on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Training target values.

        Returns
        -------
        self : LinearRegression
            Returns the fitted model to allow method chaining.
        """
        X, y = self._validate_inputs(X, y)

        if self.method == 'ols':
            self._fit_ols(X, y)
        elif self.method == 'ridge':
            self._fit_ridge(X, y)
        elif self.method == 'gd':
            self._fit_gd(X, y)

        return self

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Generate predictions for new input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix to predict on.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X @ self.weights + self.bias

    def score(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> float:
        """
        Compute the R² score on the provided data.

        R² measures how well the model explains variance in the target.
        A score of 1.0 is a perfect fit; 0.0 means the model performs
        no better than predicting the mean.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True target values.

        Returns
        -------
        float
            R² score.
        """
        if self.weights is None:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1.0 - ss_res / ss_tot)