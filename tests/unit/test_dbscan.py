"""
test_dbscan.py

Unit tests for the DBSCAN class in ml_pack.
Tests cover cluster detection, noise labeling, and input validation.
"""

import numpy as np
import pytest
from ml_pack import DBSCAN


class TestClustering:
    """Tests for basic clustering behavior."""

    def test_finds_two_clusters(self):
        """DBSCAN should identify two distinct clusters in well-separated data."""
        np.random.seed(42)
        cluster1 = np.random.randn(50, 2) + np.array([0, 0])
        cluster2 = np.random.randn(50, 2) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])

        model = DBSCAN(eps=1.0, min_samples=5).fit(X)
        n_clusters = len(set(model.labels)) - (1 if -1 in model.labels else 0)
        assert n_clusters == 2

    def test_labels_length_matches_input(self):
        """Number of labels should match number of input samples."""
        X = np.random.randn(100, 2)
        model = DBSCAN(eps=0.5, min_samples=5).fit(X)
        assert len(model.labels) == 100

    def test_fit_predict_matches_labels(self):
        """fit_predict should return the same labels as fit followed by labels."""
        X = np.random.randn(50, 2)
        model = DBSCAN(eps=0.5, min_samples=5)
        labels_from_fit_predict = model.fit_predict(X)
        model2 = DBSCAN(eps=0.5, min_samples=5)
        model2.fit(X)
        assert np.array_equal(labels_from_fit_predict, model2.labels)

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(30, 2)
        model = DBSCAN(eps=0.5, min_samples=5)
        assert model.fit(X) is model


class TestNoiseDetection:
    """Tests for noise/outlier detection."""

    def test_isolated_point_is_noise(self):
        """A point far from all others should be labeled as noise (-1)."""
        X = np.array([
            [0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.2, 0.1],
            [100, 100]  # isolated outlier
        ])
        model = DBSCAN(eps=0.5, min_samples=3).fit(X)
        assert model.labels[-1] == -1

    def test_noise_labeled_minus_one(self):
        """Noise points should always be labeled -1."""
        X = np.random.randn(50, 2)
        model = DBSCAN(eps=0.01, min_samples=20).fit(X)
        unique_labels = set(model.labels)
        # With very small eps, most points should be noise
        assert -1 in unique_labels

    def test_dense_cluster_has_no_noise(self):
        """All points in a very dense cluster should be assigned a cluster label."""
        # Tightly packed points — all should be core or border points
        X = np.random.randn(50, 2) * 0.01
        model = DBSCAN(eps=1.0, min_samples=5).fit(X)
        assert -1 not in model.labels


class TestValidation:
    """Tests for input validation and error handling."""

    def test_invalid_eps_raises(self):
        """Non-positive eps should raise ValueError."""
        with pytest.raises(ValueError):
            DBSCAN(eps=0)

    def test_invalid_min_samples_raises(self):
        """min_samples less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            DBSCAN(min_samples=0)

    def test_1d_input_raises(self):
        """Passing a 1D array should raise ValueError."""
        model = DBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]))

    def test_empty_input_raises(self):
        """Passing an empty array should raise ValueError."""
        model = DBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 2)))