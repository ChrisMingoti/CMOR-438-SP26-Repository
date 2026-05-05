"""
test_knn.py

Unit tests for the KNN class in ml_pack.
Tests cover classification behavior, prediction, scoring, and input validation.
"""

import numpy as np
import pytest
from ml_pack import KNN


class TestFitting:
    """Tests for model fitting behavior."""

    def test_fit_stores_training_data(self):
        """Training data should be stored after calling fit()."""
        X = np.random.randn(30, 4)
        y = np.array(['a'] * 15 + ['b'] * 15)
        model = KNN(k=3).fit(X, y)
        assert model.X_train is not None
        assert model.y_train is not None

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(20, 4)
        y = np.array(['a'] * 10 + ['b'] * 10)
        model = KNN(k=3)
        assert model.fit(X, y) is model


class TestPrediction:
    """Tests for prediction behavior."""

    def test_predict_shape(self):
        """Number of predictions should match number of input samples."""
        X_train = np.random.randn(50, 4)
        y_train = np.array(['a'] * 25 + ['b'] * 25)
        X_test = np.random.randn(20, 4)
        model = KNN(k=3).fit(X_train, y_train)
        assert model.predict(X_test).shape == (20,)

    def test_predicts_correct_class_on_simple_data(self):
        """KNN should correctly classify well-separated clusters."""
        X_train = np.vstack([
            np.random.randn(30, 2) + np.array([0, 0]),
            np.random.randn(30, 2) + np.array([10, 10])
        ])
        y_train = np.array(['a'] * 30 + ['b'] * 30)

        model = KNN(k=3).fit(X_train, y_train)

        # Point near cluster a
        assert model.predict(np.array([[0.1, 0.1]]))[0] == 'a'
        # Point near cluster b
        assert model.predict(np.array([[9.9, 9.9]]))[0] == 'b'

    def test_score_perfect_on_clean_data(self):
        """KNN should achieve high accuracy on well-separated data."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([10, 10])
        ])
        y = np.array(['a'] * 50 + ['b'] * 50)
        model = KNN(k=3).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_different_k_values(self):
        """Model should work correctly for various k values."""
        X = np.random.randn(50, 2)
        y = np.array(['a'] * 25 + ['b'] * 25)
        for k in [1, 3, 5, 7]:
            model = KNN(k=k).fit(X, y)
            preds = model.predict(X)
            assert len(preds) == 50


class TestValidation:
    """Tests for input validation and error handling."""

    def test_invalid_k_raises(self):
        """k less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            KNN(k=0)

    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        model = KNN(k=3)
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(10, 4))

    def test_score_before_fit_raises(self):
        """Calling score before fit should raise RuntimeError."""
        model = KNN(k=3)
        with pytest.raises(RuntimeError):
            model.score(np.random.randn(10, 4), np.array(['a'] * 10))

    def test_1d_input_raises(self):
        """Passing a 1D array as X should raise ValueError."""
        model = KNN(k=3)
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]), np.array(['a', 'b', 'c']))

    def test_mismatched_shapes_raises(self):
        """X and y with different sample counts should raise ValueError."""
        model = KNN(k=3)
        with pytest.raises(ValueError):
            model.fit(np.random.randn(10, 4), np.array(['a'] * 5))

    def test_fewer_samples_than_k_raises(self):
        """Having fewer training samples than k should raise ValueError."""
        model = KNN(k=10)
        with pytest.raises(ValueError):
            model.fit(np.random.randn(5, 4), np.array(['a'] * 5))