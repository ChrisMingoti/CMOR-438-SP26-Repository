"""
test_ensemble_methods.py

Unit tests for the RandomForest and GradientBoosting classes in ml_pack.
Tests cover training, prediction, scoring, and input validation.
"""

import numpy as np
import pytest
from ml_pack import RandomForest, GradientBoosting


class TestRandomForest:
    """Tests for the RandomForest classifier."""

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = RandomForest(n_estimators=5, random_state=42)
        assert model.fit(X, y) is model

    def test_trees_set_after_fit(self):
        """Trees list should be populated after fitting."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = RandomForest(n_estimators=10, random_state=42).fit(X, y)
        assert len(model.trees) == 10

    def test_predict_shape(self):
        """Predictions should match number of input samples."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = RandomForest(n_estimators=5, random_state=42).fit(X, y)
        assert model.predict(X).shape == (50,)

    def test_high_accuracy_on_simple_data(self):
        """Random Forest should achieve high accuracy on well-separated data."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([5, 5])
        ])
        y = np.array([0] * 50 + [1] * 50)
        model = RandomForest(n_estimators=20, max_depth=5, random_state=42).fit(X, y)
        assert model.score(X, y) > 0.90

    def test_feature_importances_set(self):
        """Feature importances should be set after fitting."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = RandomForest(n_estimators=5, random_state=42).fit(X, y)
        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == 4

    def test_invalid_n_estimators_raises(self):
        """n_estimators less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            RandomForest(n_estimators=0)

    def test_invalid_max_depth_raises(self):
        """max_depth less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            RandomForest(max_depth=0)

    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        model = RandomForest()
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(10, 4))

    def test_mismatched_shapes_raises(self):
        """X and y with different sample counts should raise ValueError."""
        model = RandomForest()
        with pytest.raises(ValueError):
            model.fit(np.random.randn(10, 4), np.zeros(5))


class TestGradientBoosting:
    """Tests for the GradientBoosting classifier."""

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GradientBoosting(n_estimators=5, random_state=42)
        assert model.fit(X, y) is model

    def test_trees_set_after_fit(self):
        """Trees list should be populated after fitting."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GradientBoosting(n_estimators=10, random_state=42).fit(X, y)
        assert len(model.trees) == 10

    def test_predict_shape(self):
        """Predictions should match number of input samples."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GradientBoosting(n_estimators=5, random_state=42).fit(X, y)
        assert model.predict(X).shape == (50,)

    def test_predict_binary_output(self):
        """Predictions should only contain 0s and 1s."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GradientBoosting(n_estimators=5, random_state=42).fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self):
        """Predicted probabilities should be between 0 and 1."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GradientBoosting(n_estimators=5, random_state=42).fit(X, y)
        probs = model.predict_proba(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_high_accuracy_on_simple_data(self):
        """Gradient Boosting should achieve high accuracy on well-separated data."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([5, 5])
        ])
        y = np.array([0] * 50 + [1] * 50)
        model = GradientBoosting(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42).fit(X, y)
        assert model.score(X, y) > 0.90

    def test_invalid_n_estimators_raises(self):
        """n_estimators less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            GradientBoosting(n_estimators=0)

    def test_invalid_learning_rate_raises(self):
        """Non-positive learning rate should raise ValueError."""
        with pytest.raises(ValueError):
            GradientBoosting(learning_rate=0)

    def test_invalid_max_depth_raises(self):
        """max_depth less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            GradientBoosting(max_depth=0)

    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        model = GradientBoosting()
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(10, 4))

    def test_mismatched_shapes_raises(self):
        """X and y with different sample counts should raise ValueError."""
        model = GradientBoosting()
        with pytest.raises(ValueError):
            model.fit(np.random.randn(10, 4), np.zeros(5))