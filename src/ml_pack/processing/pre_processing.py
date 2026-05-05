"""
pre_processing.py

Data preprocessing utilities including feature scaling and train/test splitting.
Part of the ml_pack library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Union, Sequence, Tuple, Optional


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Follows the fit/transform pattern — fit on training data only,
    then transform both train and test sets to avoid data leakage.

    Attributes
    ----------
    mean_ : np.ndarray
        Mean of each feature computed during fit.
    std_ : np.ndarray
        Standard deviation of each feature computed during fit.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and standard deviation from training data."""
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization using fitted mean and std."""
        if self.mean_ is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def train_test_split(
    *arrays,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> list:
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of array-like
        Arrays to split. Must all have the same first dimension.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int or None, default=None
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle before splitting.

    Returns
    -------
    list
        Alternating train/test splits for each input array.
    """
    if not arrays:
        raise ValueError("At least one array must be provided.")

    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same number of samples.")

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    n_test = int(np.floor(test_size * n_samples))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])

    return result