"""
test_neural_network.py

Unit tests for the NeuralNetwork class in ml_pack.
Tests cover training, prediction, scoring, and input validation.
"""

import numpy as np
import pytest
from ml_pack import NeuralNetwork


class TestTraining:
    """Tests for model training behavior."""

    def test_loss_decreases(self):
        """Training loss should decrease from first to last epoch."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        model = NeuralNetwork(hidden_layers=[16, 8], learning_rate=0.01, epochs=500, random_state=42)
        model.fit(X, y)
        assert model.loss_history[0] > model.loss_history[-1]

    def test_loss_history_length(self):
        """Loss history should have one entry per epoch."""
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50).astype(float)
        model = NeuralNetwork(hidden_layers=[8], epochs=200, random_state=0).fit(X, y)
        assert len(model.loss_history) == 200

    def test_fit_returns_self(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(30, 4)
        y = np.random.randint(0, 2, 30).astype(float)
        model = NeuralNetwork(hidden_layers=[8], random_state=0)
        assert model.fit(X, y) is model

    def test_params_set_after_fit(self):
        """params should not be None after training."""
        X = np.random.randn(30, 4)
        y = np.random.randint(0, 2, 30).astype(float)
        model = NeuralNetwork(hidden_layers=[8], random_state=0).fit(X, y)
        assert model.params is not None


class TestPrediction:
    """Tests for prediction behavior."""

    def test_predict_shape(self):
        """Predictions should match the number of input samples."""
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50).astype(float)
        model = NeuralNetwork(hidden_layers=[8], epochs=100, random_state=0).fit(X, y)
        assert model.predict(X).shape == (50,)

    def test_predict_binary_output(self):
        """Predictions should only contain 0s and 1s."""
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50).astype(float)
        model = NeuralNetwork(hidden_layers=[8], epochs=100, random_state=0).fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_range(self):
        """Predicted probabilities should be between 0 and 1."""
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50).astype(float)
        model = NeuralNetwork(hidden_layers=[8], epochs=100, random_state=0).fit(X, y)
        probs = model.predict_proba(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_high_accuracy_on_simple_data(self):
        """Network should achieve high accuracy on linearly separable data."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)
        model = NeuralNetwork(hidden_layers=[16, 8], learning_rate=0.01, epochs=1000, random_state=42)
        model.fit(X, y)
        assert model.score(X, y) > 0.90


class TestValidation:
    """Tests for input validation and error handling."""

    def test_predict_before_fit_raises(self):
        """Calling predict before fit should raise RuntimeError."""
        model = NeuralNetwork()
        with pytest.raises(RuntimeError):
            model.predict(np.random.randn(10, 4))

    def test_score_before_fit_raises(self):
        """Calling score before fit should raise RuntimeError."""
        model = NeuralNetwork()
        with pytest.raises(RuntimeError):
            model.score(np.random.randn(10, 4), np.zeros(10))

    def test_invalid_learning_rate_raises(self):
        """Non-positive learning rate should raise ValueError."""
        with pytest.raises(ValueError):
            NeuralNetwork(learning_rate=0)

    def test_invalid_epochs_raises(self):
        """Epochs less than 1 should raise ValueError."""
        with pytest.raises(ValueError):
            NeuralNetwork(epochs=0)

    def test_empty_hidden_layers_raises(self):
        """Empty hidden layers list should raise ValueError."""
        with pytest.raises(ValueError):
            NeuralNetwork(hidden_layers=[])

    def test_mismatched_shapes_raises(self):
        """X and y with different sample counts should raise ValueError."""
        model = NeuralNetwork()
        with pytest.raises(ValueError):
            model.fit(np.random.randn(10, 4), np.zeros(5))

    def test_1d_input_raises(self):
        """Passing a 1D array as X should raise ValueError."""
        model = NeuralNetwork()
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]), np.array([0, 1, 0]))