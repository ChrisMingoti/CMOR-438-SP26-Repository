"""
ensemble_methods.py

This module implements ensemble learning methods from scratch using NumPy.
Includes Random Forest and Gradient Boosting classifiers. Both build on
the DecisionTree implementation in ml_pack. Part of the ml_pack supervised
learning library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence, List
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    """
    A from-scratch implementation of the Random Forest classifier.

    Random Forest builds an ensemble of decision trees, each trained on a
    random bootstrap sample of the data and a random subset of features.
    Final predictions are made by majority vote across all trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees to build.
    max_depth : int or None, default=None
        Maximum depth of each tree.
    max_features : int or None, default=None
        Number of features to consider at each split. If None, uses
        the square root of the total number of features.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    trees : list of DecisionTree
        Fitted decision trees. Set after calling fit().
    feature_importances_ : np.ndarray or None
        Average feature importances across all trees.

    Examples
    --------
    >>> model = RandomForest(n_estimators=100, max_depth=5, random_state=42)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees: List[DecisionTree] = []
        self.feature_indices_: List[np.ndarray] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "RandomForest":
        """
        Train the Random Forest on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Training labels.

        Returns
        -------
        self : RandomForest
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        max_features = self.max_features or int(np.sqrt(n_features))
        importances = np.zeros(n_features)

        self.trees = []
        self.feature_indices_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # Random feature subset
            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices_.append(feature_idx)

            # Train a decision tree on the bootstrap sample
            tree = DecisionTree(max_depth=self.max_depth, random_state=None)
            tree.fit(X_sample[:, feature_idx], y_sample)
            self.trees.append(tree)

            # Accumulate feature importances
            for i, fi in enumerate(feature_idx):
                importances[fi] += tree.feature_importances_[i]

        # Normalize feature importances
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

        return self

    def predict(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Predict class labels by majority vote across all trees.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix to predict on.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if not self.trees:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)

        # Collect predictions from all trees
        all_preds = np.array([
            tree.predict(X[:, feature_idx])
            for tree, feature_idx in zip(self.trees, self.feature_indices_)
        ])

        # Majority vote for each sample
        return np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])

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
        if not self.trees:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y)
        return float(np.mean(y == self.predict(X)))


class GradientBoosting:
    """
    A from-scratch implementation of Gradient Boosting for binary classification.

    Gradient Boosting builds trees sequentially where each tree is trained to
    correct the residual errors of the previous ensemble. Uses decision stumps
    (shallow trees) as weak learners and log-loss as the objective function.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages.
    learning_rate : float, default=0.1
        Shrinkage factor applied to each tree's contribution.
    max_depth : int, default=3
        Maximum depth of each weak learner tree.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    trees : list of DecisionTree
        Fitted weak learner trees.
    initial_prediction : float
        Initial log-odds prediction before boosting.

    Examples
    --------
    >>> model = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = None
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.trees: List[DecisionTree] = []
        self.initial_prediction: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "GradientBoosting":
        """
        Train the Gradient Boosting model on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Binary target labels (0 or 1).

        Returns
        -------
        self : GradientBoosting
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize with log-odds of the mean
        mean = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        self.initial_prediction = np.log(mean / (1 - mean))

        F = np.full(len(y), self.initial_prediction)
        self.trees = []

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradient of log-loss)
            p = self._sigmoid(F)
            residuals = y - p

            # Fit a shallow tree to the residuals
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            F += self.learning_rate * tree.predict(X).astype(float)

        return self

    def predict_proba(self, X: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Return predicted probabilities for the positive class.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted probabilities between 0 and 1.
        """
        if not self.trees:
            raise RuntimeError("Model must be fitted before calling predict_proba().")

        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X).astype(float)

        return self._sigmoid(F)

    def predict(self, X: Union[np.ndarray, Sequence], threshold: float = 0.5) -> np.ndarray:
        """
        Return binary class predictions.

        Parameters
        ----------
        X : array-like
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
            True labels.

        Returns
        -------
        float
            Proportion of correctly classified samples.
        """
        if not self.trees:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y)
        return float(np.mean(y == self.predict(X)))