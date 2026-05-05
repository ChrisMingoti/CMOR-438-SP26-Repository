"""
test_linear_regression.py

Unit tests for the LinearRegression class in ml_pack.
Validates correctness of all three solvers: OLS, Ridge, and Gradient Descent,
as well as input validation and error handling.
"""

import numpy as np
import pytest
from ml_pack import LinearRegression


class TestOLS:
    """Tests for the Ordinary Least Squares solver."""

    def test_recovers_exact_slope(self):
        """OLS should recover the exact slope on perfectly linear data."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        model = LinearRegression(method='ols').fit(X, y)
        assert np.isclose(model.weights[0], 2.0, atol=1e-6)
        assert np.isclose(model.bias, 0.0, atol=1e-6)

    def test_recovers_nonzero_intercept(self):
        """OLS should correctly identify a nonzero bias term."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([3, 5, 7, 9], dtype=float)  # y = 2x + 1
        model = LinearRegression(method='ols').fit(X, y)
        assert np.isclose(model.weights[0], 2.0, atol=1e-6)
        assert np.isclose(model.bias, 1.0, atol=1e-6)

    def test_multivariate_recovery(self):
        """OLS should recover true weights in the multivariate case."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_weights = np.array([1.5, -2.0, 0.5])
        y = X @ true_weights + 3.0
        model = LinearRegression(method='ols').fit(X, y)
        assert np.allclose(model.weights, true_weights, atol=1e-6)
        assert np.isclose(model.bias, 3.0, atol=1e-6)

    def test_output_shape(self):
        """Number of predictions should match number of input samples."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        model = LinearRegression(method='ols').fit(X, y)
        assert model.predict(X).shape == (50,)

    def test_perfect_r2(self):
        """R² should equal 1.0 when the model fits the data exactly."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        model = LinearRegression(method='ols').fit(X, y)
        assert np.isclose(model.score(X, y), 1.0, atol=1e-6)

    def test_method_chaining(self):
        """fit() should return the model instance to support method chaining."""
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        model = LinearRegression(method='ols')
        assert model.fit(X, y) is model

    def test_parameters_set_after_fit(self):
        """Weights and bias should not be None after training."""
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        model = LinearRegression(method='ols').fit(X, y)
        assert model.weights is not None
        assert model.bias is not None


class TestRidge:
    """Tests for the Ridge regression solver."""

    def test_regularization_shrinks_weights(self):
        """Ridge should produce smaller weights than OLS when alpha is large."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        ols = LinearRegression(method='ols').fit(X, y)
        ridge = LinearRegression(method='ridge', alpha=100.0).fit(X, y)
        assert np.linalg.norm(ridge.weights) < np.linalg.norm(ols.weights)

    def test_larger_alpha_shrinks_more(self):
        """Increasing alpha should further reduce the magnitude of weights."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        low_reg = LinearRegression(method='ridge', alpha=0.1).fit(X, y)
        high_reg = LinearRegression(method='ridge', alpha=100.0).fit(X, y)
        assert np.linalg.norm(high_reg.weights) < np.linalg.norm(low_reg.weights)

    def test_reasonable_r2(self):
        """Ridge should achieve a strong R² score on clean linear data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1.5, -2.0, 0.5]) + 0.1
        model = LinearRegression(method='ridge', alpha=0.1).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_output_shape(self):
        """Ridge predictions should match the number of input samples."""
        X = np.random.randn(40, 3)
        y = np.random.randn(40)
        model = LinearRegression(method='ridge', alpha=1.0).fit(X, y)
        assert model.predict(X).shape == (40,)


class TestGradientDescent:
    """Tests for the Gradient Descent solver."""

    def test_loss_decreases(self):
        """Training loss should decrease from the first to the last epoch."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X @ np.array([2.0, -1.0]) + 0.5
        model = LinearRegression(method='gd', learning_rate=0.01, epochs=500, random_state=42)
        model.fit(X, y)
        assert model.cost_history[0] > model.cost_history[-1]

    def test_approximately_recovers_weights(self):
        """GD should approximately recover the true weights given enough epochs."""
        X = np.array([[1], [2], [3], [4]], dtype=float)
        y = np.array([2, 4, 6, 8], dtype=float)
        model = LinearRegression(method='gd', learning_rate=0.1, epochs=1000, random_state=0)
        model.fit(X, y)
        assert np.isclose(model.weights[0], 2.0, atol=0.1)
        assert np.isclose(model.bias, 0.0, atol=0.1)

    def test_output_shape(self):
        """GD predictions should match the number of input samples."""
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        model = LinearRegression(method='gd', epochs=100, random_state=0).fit(X, y)
        assert model.predict(X).shape == (50,)

    def test_cost_history_length(self):
        """Cost history should contain exactly one entry per epoch."""
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        model = LinearRegression(method='gd', epochs=200, random_state=1).fit(X, y)
        assert len(model.cost_history) == 200


class TestValidation:
    """Tests for input validation and error handling."""

    def test_predict_before_fit(self):
        """Calling predict on an unfitted model should raise RuntimeError."""
        model = LinearRegression()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1, 2]]))

    def test_score_before_fit(self):
        """Calling score on an unfitted model should raise RuntimeError."""
        model = LinearRegression()
        with pytest.raises(RuntimeError):
            model.score(np.array([[1, 2]]), np.array([1]))

    def test_invalid_method(self):
        """An unrecognized method name should raise ValueError."""
        with pytest.raises(ValueError):
            LinearRegression(method='invalid')

    def test_negative_alpha(self):
        """A negative regularization strength should raise ValueError."""
        with pytest.raises(ValueError):
            LinearRegression(method='ridge', alpha=-1.0)

    def test_sample_count_mismatch(self):
        """X and y with different numbers of rows should raise ValueError."""
        model = LinearRegression()
        X = np.random.randn(10, 2)
        y = np.random.randn(5)
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_1d_input_raises(self):
        """Passing a 1D array as X should raise ValueError."""
        model = LinearRegression()
        with pytest.raises(ValueError):
            model.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))