"""
decision_tree.py

This module implements a Decision Tree classifier from scratch using NumPy.
It uses the CART algorithm with Gini impurity for splitting. Part of the
ml_pack supervised learning library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Optional, Union, Sequence
from collections import Counter


class Node:
    """
    Represents a single node in the decision tree.

    Attributes
    ----------
    feature : int or None
        Index of the feature used to split at this node.
    threshold : float or None
        Value used to split the feature.
    left : Node or None
        Left subtree (feature <= threshold).
    right : Node or None
        Right subtree (feature > threshold).
    value : any
        Predicted class label if this is a leaf node.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """Returns True if this node is a leaf node."""
        return self.value is not None


class DecisionTree:
    """
    A from-scratch implementation of a Decision Tree classifier.

    Uses the CART algorithm with Gini impurity to find the best splits.
    Supports multiclass classification and controls overfitting through
    max_depth and min_samples_split parameters.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. None means nodes are expanded until
        all leaves are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    random_state : int or None, default=None
        Seed for reproducibility.

    Attributes
    ----------
    root : Node or None
        Root node of the fitted tree. Set after calling fit().
    n_classes : int or None
        Number of unique classes in the training data.
    feature_importances : np.ndarray or None
        Importance of each feature based on Gini impurity reduction.

    Examples
    --------
    >>> model = DecisionTree(max_depth=5, random_state=42)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> print(model.score(X_test, y_test))
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: Optional[int] = None
    ):
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.root: Optional[Node] = None
        self.n_classes: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None

    def _gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity for a set of labels.

        Gini impurity measures how often a randomly chosen element
        would be incorrectly classified. A pure node has Gini = 0.
        """
        n = len(y)
        if n == 0:
            return 0.0
        counts = Counter(y)
        return 1.0 - sum((count / n) ** 2 for count in counts.values())

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the best feature and threshold to split on using Gini impurity.

        Returns
        -------
        best_feature : int
            Index of the best feature to split on.
        best_threshold : float
            Threshold value for the best split.
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        n = len(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                # Weighted Gini impurity of the split
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (left_mask.sum() / n) * gini_left + \
                                (right_mask.sum() / n) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for the current node.
        y : np.ndarray
            Labels for the current node.
        depth : int
            Current depth in the tree.

        Returns
        -------
        Node
            The root node of the subtree.
        """
        n_samples = len(y)
        n_labels = len(np.unique(y))

        # Stopping conditions
        if (
            n_labels == 1 or
            n_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth)
        ):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        # Track feature importance
        self.feature_importances_[best_feature] += self._gini(y) * n_samples

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def fit(self, X: Union[np.ndarray, Sequence], y: Union[np.ndarray, Sequence]) -> "DecisionTree":
        """
        Train the decision tree on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Training labels.

        Returns
        -------
        self : DecisionTree
            Returns the fitted model to allow method chaining.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self._n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.feature_importances_ = np.zeros(self._n_features)

        self.root = self._build_tree(X, y)

        # Normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def _predict_single(self, x: np.ndarray, node: Node):
        """Traverse the tree to predict the class for a single sample."""
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

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
        if self.root is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)
        return np.array([self._predict_single(x, self.root) for x in X])

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
        if self.root is None:
            raise RuntimeError("Model must be fitted before calling score().")

        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y == y_pred))