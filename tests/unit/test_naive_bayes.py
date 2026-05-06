"""
test_naive_bayes.py

Unit tests for the GaussianNaiveBayes class in ml_pack.
Tests cover training, prediction, scoring, and input validation.
"""

import numpy as np
import pytest
from ml_pack import GaussianNaiveBayes


class TestTraining:
    """Tests for model training behavior."""

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(50, 4)
        y = np.array(['a'] * 25 + ['b'] * 25)
        model = GaussianNaiveBayes()
        assert model.fit(X, y) is model

    def test_classes_set_after_fit(self):
        """Classes should be set after fitting."""
        X = np.random.randn(50, 4)
        y = np.array(['a'] * 25 + ['b'] * 25)
        model = GaussianNaiveBayes().fit(X, y)
        assert model.classes is not None
        assert set(model.classes) == {'a', 'b'}

    def test_means_shape(self):
        """Means should have shape (n_classes, n_features)."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GaussianNaiveBayes().fit(X, y)
        assert model.means.shape == (2, 4)

    def test_variances_shape(self):
        """Variances should have shape (n_classes, n_features)."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GaussianNaiveBayes().fit(X, y)
        assert model.variances.shape == (2, 4)

    def test_priors_sum_to_one(self):
        """Class priors should sum to 1."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GaussianNaiveBayes().fit(X, y)
        assert np.isclose(model.priors.sum(), 1.0, atol=1e-6)


class TestPrediction:
    """Tests for prediction behavior."""

    def test_predict_shape(self):
        """Predictions should match number of input samples."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GaussianNaiveBayes().fit(X, y)
        assert model.predict(X).shape == (50,)

    def test_predict_proba_shape(self):
        """Predicted probabilities should have shape (n_samples, n_classes)."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GaussianNaiveBayes().fit(X, y)
        assert model.predict_proba(X).shape == (50, 2)

    def test_predict_proba_sums_to_one(self):
        """Predicted probabilities should sum to 1 for each sample."""
        X = np.random.randn(50, 4)
        y = np.array([0] * 25 + [1] * 25)
        model = GaussianNaiveBayes().fit(X, y)
        probs = model.predict_proba(X)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_high_accuracy_on_simple_data(self):
        """Model should achieve high accuracy on well-separated data."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([5, 5])
        ])
        y = np.array([0] * 50 + [1] * 50)
        model = GaussianNaiveBayes().fit(X, y)
        assert model.score(X, y) > 0.95

    def test_multiclass_prediction(self):
        """Model should handle multiclass classification."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + np.array([0, 0]),
            np.random.randn(30, 2) + np.array([5, 5]),
            np.random.randn(30, 2) + np.array([10, 0])
        ])
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        model = GaussianNaiveBayes().fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2})


class TestValidation:
    """Tests for input validation and error handling."""

    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        model = GaussianNaiveBayes()
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(10, 4))

    def test_predict_proba_before_fit_raises(self):
        """Calling predict_proba before fit should raise RuntimeError."""
        model = GaussianNaiveBayes()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.random.randn(10, 4))

    def test_score_before_fit_raises(self):
        """Calling score before fit should raise RuntimeError."""
        model = GaussianNaiveBayes()
        with pytest.raises(RuntimeError):
            model.score(np.random.randn(10, 4), np.zeros(10))

    def test_1d_input_raises(self):
        """Passing a 1D array as X should raise ValueError."""
        model = GaussianNaiveBayes()
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]), np.array([0, 1, 0]))

    def test_mismatched_shapes_raises(self):
        """X and y with different sample counts should raise ValueError."""
        model = GaussianNaiveBayes()
        with pytest.raises(ValueError):
            model.fit(np.random.randn(10, 4), np.zeros(5))