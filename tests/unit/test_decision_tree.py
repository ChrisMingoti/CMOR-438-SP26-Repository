"""
test_decision_tree.py

Unit tests for the DecisionTree class in ml_pack.
Tests cover classification behavior, feature importance, and input validation.
"""

import numpy as np
import pytest
from ml_pack import DecisionTree


class TestTraining:
    """Tests for model training behavior."""

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(30, 4)
        y = np.array([0] * 15 + [1] * 15)
        model = DecisionTree(max_depth=3)
        assert model.fit(X, y) is model

    def test_root_set_after_fit(self):
        """Root node should be set after fitting."""
        X = np.random.randn(30, 4)
        y = np.array([0] * 15 + [1] * 15)
        model = DecisionTree(max_depth=3).fit(X, y)
        assert model.root is not None

    def test_feature_importances_set_after_fit(self):
        """Feature importances should be set after fitting."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = DecisionTree(max_depth=3).fit(X, y)
        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == 4

    def test_feature_importances_sum_to_one(self):
        """Feature importances should sum to 1."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = DecisionTree(max_depth=3).fit(X, y)
        assert np.isclose(model.feature_importances_.sum(), 1.0, atol=1e-6)


class TestPrediction:
    """Tests for prediction behavior."""

    def test_predict_shape(self):
        """Number of predictions should match number of input samples."""
        X_train = np.random.randn(50, 4)
        y_train = np.array([0] * 25 + [1] * 25)
        X_test = np.random.randn(20, 4)
        model = DecisionTree(max_depth=3).fit(X_train, y_train)
        assert model.predict(X_test).shape == (20,)

    def test_perfect_fit_on_clean_data(self):
        """Tree should achieve perfect accuracy on linearly separable data."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        model = DecisionTree(max_depth=5).fit(X, y)
        assert model.score(X, y) == 1.0

    def test_high_accuracy_on_simple_data(self):
        """Tree should achieve high accuracy on well-separated clusters."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([5, 5])
        ])
        y = np.array([0] * 50 + [1] * 50)
        model = DecisionTree(max_depth=5).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_multiclass_prediction(self):
        """Tree should handle multiclass classification."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + np.array([0, 0]),
            np.random.randn(30, 2) + np.array([5, 5]),
            np.random.randn(30, 2) + np.array([10, 0])
        ])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        model = DecisionTree(max_depth=5).fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2})


class TestValidation:
    """Tests for input validation and error handling."""

    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        model = DecisionTree()
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(10, 4))

    def test_score_before_fit_raises(self):
        """Calling score before fit should raise RuntimeError."""
        model = DecisionTree()
        with pytest.raises(RuntimeError):
            model.score(np.random.randn(10, 4), np.zeros(10))

    def test_invalid_max_depth_raises(self):
        """max_depth less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            DecisionTree(max_depth=0)

    def test_invalid_min_samples_split_raises(self):
        """min_samples_split less than 2 should raise ValueError."""
        with pytest.raises(ValueError):
            DecisionTree(min_samples_split=1)

    def test_1d_input_raises(self):
        """Passing a 1D array as X should raise ValueError."""
        model = DecisionTree()
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]), np.array([0, 1, 0]))

    def test_mismatched_shapes_raises(self):
        """X and y with different sample counts should raise ValueError."""
        model = DecisionTree()
        with pytest.raises(ValueError):
            model.fit(np.random.randn(10, 4), np.zeros(5))