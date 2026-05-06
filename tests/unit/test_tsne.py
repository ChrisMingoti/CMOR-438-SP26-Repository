"""
test_tsne.py

Unit tests for the TSNE class in ml_pack.
Tests cover embedding shape, input validation, and basic behavior.
"""

import numpy as np
import pytest
from ml_pack import TSNE


class TestFitting:
    """Tests for t-SNE fitting behavior."""

    def test_fit_transform_shape(self):
        """Output embedding should have shape (n_samples, n_components)."""
        X = np.random.randn(50, 10)
        model = TSNE(n_components=2, n_iter=100, random_state=42)
        X_reduced = model.fit_transform(X)
        assert X_reduced.shape == (50, 2)

    def test_embedding_set_after_fit(self):
        """Embedding should be set after calling fit()."""
        X = np.random.randn(30, 5)
        model = TSNE(n_components=2, n_iter=50, random_state=0)
        model.fit(X)
        assert model.embedding is not None

    def test_fit_transform_matches_fit(self):
        """fit_transform should produce the same embedding as fit."""
        X = np.random.randn(30, 5)
        model1 = TSNE(n_components=2, n_iter=50, random_state=0)
        result1 = model1.fit_transform(X)
        model2 = TSNE(n_components=2, n_iter=50, random_state=0)
        model2.fit(X)
        assert np.allclose(result1, model2.embedding, atol=1e-6)

    def test_different_n_components(self):
        """Should work for different n_components values."""
        X = np.random.randn(30, 10)
        for n in [1, 2, 3]:
            model = TSNE(n_components=n, n_iter=50, random_state=0)
            result = model.fit_transform(X)
            assert result.shape == (30, n)

    def test_embedding_centered(self):
        """Embedding should be approximately centered at zero."""
        X = np.random.randn(50, 10)
        model = TSNE(n_components=2, n_iter=200, random_state=42)
        X_reduced = model.fit_transform(X)
        assert np.allclose(X_reduced.mean(axis=0), 0, atol=1e-6)


class TestValidation:
    """Tests for input validation and error handling."""

    def test_invalid_n_components_raises(self):
        """n_components less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            TSNE(n_components=0)

    def test_invalid_perplexity_raises(self):
        """Non-positive perplexity should raise ValueError."""
        with pytest.raises(ValueError):
            TSNE(perplexity=0)

    def test_invalid_learning_rate_raises(self):
        """Non-positive learning rate should raise ValueError."""
        with pytest.raises(ValueError):
            TSNE(learning_rate=0)

    def test_invalid_n_iter_raises(self):
        """n_iter less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            TSNE(n_iter=0)

    def test_1d_input_raises(self):
        """Passing a 1D array should raise ValueError."""
        model = TSNE(n_components=2)
        with pytest.raises(ValueError):
            model.fit_transform(np.array([1, 2, 3]))

    def test_single_sample_raises(self):
        """Passing only one sample should raise ValueError."""
        model = TSNE(n_components=2)
        with pytest.raises(ValueError):
            model.fit_transform(np.random.randn(1, 5))