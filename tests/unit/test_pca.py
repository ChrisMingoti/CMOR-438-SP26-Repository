"""
test_pca.py

Unit tests for the PCA class in ml_pack.
Tests cover dimensionality reduction, explained variance, and input validation.
"""

import numpy as np
import pytest
from ml_pack import PCA


class TestFitting:
    """Tests for model fitting behavior."""

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(50, 10)
        model = PCA(n_components=3)
        assert model.fit(X) is model

    def test_components_shape(self):
        """Components should have shape (n_components, n_features)."""
        X = np.random.randn(50, 10)
        model = PCA(n_components=3).fit(X)
        assert model.components.shape == (3, 10)

    def test_explained_variance_ratio_sums_to_one(self):
        """Explained variance ratio across all components should sum to 1."""
        X = np.random.randn(50, 5)
        model = PCA(n_components=5).fit(X)
        assert np.isclose(model.explained_variance_ratio_.sum(), 1.0, atol=1e-6)

    def test_explained_variance_decreasing(self):
        """Explained variance should decrease across components."""
        X = np.random.randn(100, 10)
        model = PCA(n_components=5).fit(X)
        for i in range(len(model.explained_variance) - 1):
            assert model.explained_variance[i] >= model.explained_variance[i + 1]

    def test_mean_computed_correctly(self):
        """Mean should match the per-feature mean of the training data."""
        X = np.random.randn(50, 5)
        model = PCA(n_components=3).fit(X)
        assert np.allclose(model.mean, np.mean(X, axis=0), atol=1e-6)


class TestTransform:
    """Tests for data transformation."""

    def test_transform_shape(self):
        """Transformed data should have shape (n_samples, n_components)."""
        X = np.random.randn(50, 10)
        model = PCA(n_components=3).fit(X)
        assert model.transform(X).shape == (50, 3)

    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform should produce the same result as fit then transform."""
        X = np.random.randn(50, 10)
        model1 = PCA(n_components=3)
        result1 = model1.fit_transform(X)
        model2 = PCA(n_components=3)
        model2.fit(X)
        result2 = model2.transform(X)
        assert np.allclose(np.abs(result1), np.abs(result2), atol=1e-6)

    def test_inverse_transform_shape(self):
        """Inverse transformed data should have original shape."""
        X = np.random.randn(50, 10)
        model = PCA(n_components=3).fit(X)
        X_reduced = model.transform(X)
        X_reconstructed = model.inverse_transform(X_reduced)
        assert X_reconstructed.shape == X.shape

    def test_reconstruction_error_decreases_with_more_components(self):
        """More components should result in lower reconstruction error."""
        X = np.random.randn(100, 10)

        model_2 = PCA(n_components=2).fit(X)
        X_reconstructed_2 = model_2.inverse_transform(model_2.transform(X))
        error_2 = np.mean((X - X_reconstructed_2) ** 2)

        model_8 = PCA(n_components=8).fit(X)
        X_reconstructed_8 = model_8.inverse_transform(model_8.transform(X))
        error_8 = np.mean((X - X_reconstructed_8) ** 2)

        assert error_8 < error_2


class TestValidation:
    """Tests for input validation and error handling."""

    def test_transform_before_fit_raises(self):
        """Calling transform before fit should raise RuntimeError."""
        model = PCA(n_components=3)
        with pytest.raises(RuntimeError):
            model.transform(np.random.randn(10, 5))

    def test_inverse_transform_before_fit_raises(self):
        """Calling inverse_transform before fit should raise RuntimeError."""
        model = PCA(n_components=3)
        with pytest.raises(RuntimeError):
            model.inverse_transform(np.random.randn(10, 3))

    def test_invalid_n_components_raises(self):
        """n_components less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            PCA(n_components=0)

    def test_n_components_exceeds_features_raises(self):
        """n_components greater than n_features should raise ValueError."""
        model = PCA(n_components=20)
        with pytest.raises(ValueError):
            model.fit(np.random.randn(50, 10))

    def test_1d_input_raises(self):
        """Passing a 1D array should raise ValueError."""
        model = PCA(n_components=2)
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]))

    def test_single_sample_raises(self):
        """Passing only one sample should raise ValueError."""
        model = PCA(n_components=2)
        with pytest.raises(ValueError):
            model.fit(np.random.randn(1, 5))