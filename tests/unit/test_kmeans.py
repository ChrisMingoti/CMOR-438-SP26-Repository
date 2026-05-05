"""
test_kmeans.py

Unit tests for the KMeans class in ml_pack.
Tests cover clustering behavior, convergence, and input validation.
"""

import numpy as np
import pytest
from ml_pack import KMeans


class TestClustering:
    """Tests for basic clustering behavior."""

    def test_correct_number_of_clusters(self):
        """KMeans should produce exactly n_clusters unique labels."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([10, 0]),
            np.random.randn(50, 2) + np.array([5, 10])
        ])
        model = KMeans(n_clusters=3, random_state=42).fit(X)
        assert len(np.unique(model.labels)) == 3

    def test_labels_length_matches_input(self):
        """Number of labels should match number of input samples."""
        X = np.random.randn(100, 2)
        model = KMeans(n_clusters=3, random_state=0).fit(X)
        assert len(model.labels) == 100

    def test_fit_predict_matches_labels(self):
        """fit_predict should return the same labels as fit."""
        X = np.random.randn(50, 2)
        model1 = KMeans(n_clusters=3, random_state=0)
        labels1 = model1.fit_predict(X)
        model2 = KMeans(n_clusters=3, random_state=0)
        model2.fit(X)
        assert np.array_equal(labels1, model2.labels)

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(30, 2)
        model = KMeans(n_clusters=3, random_state=0)
        assert model.fit(X) is model

    def test_centroids_shape(self):
        """Centroids should have shape (n_clusters, n_features)."""
        X = np.random.randn(100, 4)
        model = KMeans(n_clusters=5, random_state=0).fit(X)
        assert model.centroids.shape == (5, 4)

    def test_inertia_is_positive(self):
        """Inertia should always be a positive number."""
        X = np.random.randn(50, 2)
        model = KMeans(n_clusters=3, random_state=0).fit(X)
        assert model.inertia > 0

    def test_predict_assigns_to_nearest_centroid(self):
        """predict() should assign new points to the nearest centroid."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([0, 0]),
            np.random.randn(50, 2) + np.array([10, 10])
        ])
        model = KMeans(n_clusters=2, random_state=42).fit(X)

        # A point near [0,0] should get a different label than one near [10,10]
        p1 = model.predict(np.array([[0.1, 0.1]]))
        p2 = model.predict(np.array([[9.9, 9.9]]))
        assert p1[0] != p2[0]


class TestConvergence:
    """Tests for convergence behavior."""

    def test_n_iter_set_after_fit(self):
        """n_iter should be set after fitting."""
        X = np.random.randn(50, 2)
        model = KMeans(n_clusters=3, random_state=0).fit(X)
        assert model.n_iter > 0

    def test_more_iterations_same_or_less_inertia(self):
        """More iterations should produce equal or lower inertia."""
        X = np.random.randn(100, 2)
        model_low = KMeans(n_clusters=3, max_iter=5, random_state=42).fit(X)
        model_high = KMeans(n_clusters=3, max_iter=300, random_state=42).fit(X)
        assert model_high.inertia <= model_low.inertia


class TestValidation:
    """Tests for input validation and error handling."""

    def test_invalid_n_clusters_raises(self):
        """n_clusters less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            KMeans(n_clusters=0)

    def test_invalid_max_iter_raises(self):
        """max_iter less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            KMeans(max_iter=0)

    def test_invalid_tol_raises(self):
        """Negative tol should raise ValueError."""
        with pytest.raises(ValueError):
            KMeans(tol=-1)

    def test_1d_input_raises(self):
        """Passing a 1D array should raise ValueError."""
        model = KMeans(n_clusters=2)
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]))

    def test_fewer_samples_than_clusters_raises(self):
        """Having fewer samples than clusters should raise ValueError."""
        model = KMeans(n_clusters=10)
        with pytest.raises(ValueError):
            model.fit(np.random.randn(5, 2))

    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        model = KMeans(n_clusters=3)
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(10, 2))