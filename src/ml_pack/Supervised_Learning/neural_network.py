"""
neural_network.py

This module implements a feedforward neural network (multilayer perceptron)
from scratch using NumPy. It supports binary and multiclass classification
with configurable hidden layers, ReLU activations, and sigmoid/softmax output.
Part of the ml_pack supervised learning library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import List, Optional, Union, Sequence


class NeuralNetwork:
    """
    A from-scratch feedforward neural network supporting binary classification.

    The network uses:
        - He initialization for weights
        - ReLU activation for hidden layers
        - Sigmoid activation for the output layer
        - Binary cross-entropy loss
        - Batch gradient descent for optimization

    Parameters
    ----------
    hidden_layers : list of int
        Number of neurons in each hidden layer.
        e.g. [64, 32] creates two hidden layers with 64 and 32 neurons.
    learning_rate : float, default=0.01
        Step size for gradient descent weight updates.
    epochs : int, default=1000
        Number of full passes through the training data.
    random_state : int or None, default=None
        Seed for reproducible weight initialization.

    Attributes
    ----------
    params : list of dict or None
        Learned weights and biases for each layer. Set after calling fit().
    loss_history : list of float
        Training loss recorded at each epoch.

    Examples
    --------
    >>> model = NeuralNetwork(hidden_layers=[64, 32], learning_rate=0.01, epochs=1000)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        random_state: Optional[int] = None
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer.")

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.params = None
        self.loss_history: List[float] = []
        self._n_features: Optional[int] = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation — maps input to (0, 1)."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation — zeroes out negative values."""
        return np.maximum(0, z)

    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU — 1 for positive inputs, 0 otherwise."""
        return (z > 0).astype(float)

    def _initialize_weights(self, layer_sizes: List[int]) -> List[dict]:
        """
        Initialize weights using He initialization and biases as zeros.

        He initialization scales weights by sqrt(2 / n_inputs), which works
        well with ReLU activations to maintain variance across layers.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        params = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            params.append({'W': W, 'b': b})
        return params

    def _forward(self, X: np.ndarray):
        """
        Run a forward pass through all layers.

        Returns activations and pre-activations for use in backpropagation.
        """
        activations = [X]
        pre_activations = []

        for i, layer in enumerate(self.params):
            z = activations[-1] @ layer['W'] + layer['b']
            pre_activations.append(z)

            # Hidden layers use ReLU, output layer uses sigmoid
            if i < len(self.params) - 1:
                a = self._relu(z)
            else:
                a = self._sigmoid(z)

            activations.append(a)

        return activations, pre_activations

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def _backward(self, X: np.ndarray, y: np.ndarray, activations: list, pre_activations: list) -> None:
        """
        Run backpropagation and update weights and biases.

        Computes gradients layer by layer using the chain rule,
        then applies gradient descent updates.
        """
        n = X.shape[0]
        y = y.reshape(-1, 1)

        # Start with output layer error
        delta = activations[-1] - y

        for i in reversed(range(len(self.params))):
            dW = activations[i].T @ delta / n
            db = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                delta = delta @ self.params[i]['W'].T * self._relu_derivative(pre_activations[i - 1])

            self.params[i]['W'] -= self.learning_rate * dW
            self.params[i]['b'] -= self.learning_rate * db

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "NeuralNetwork":
        """
        Train the neural network on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : NeuralNetwork
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self._n_features = X.shape[1]
        layer_sizes = [self._n_features] + self.hidden_layers + [1]
        self.params = self._initialize_weights(layer_sizes)
        self.loss_history = []

        for _ in range(self.epochs):
            activations, pre_activations = self._forward(X)
            loss = self._compute_loss(y, activations[-1].ravel())
            self.loss_history.append(loss)
            self._backward(X, y, activations, pre_activations)

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
        if self.params is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        X = np.asarray(X, dtype=float)
        activations, _ = self._forward(X)
        return activations[-1].ravel()

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
        if self.params is None:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return float(np.mean(y == y_pred))