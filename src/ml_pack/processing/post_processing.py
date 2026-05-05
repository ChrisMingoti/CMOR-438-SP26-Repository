"""
post_processing.py

Evaluation metrics for regression and classification models.
Part of the ml_pack library developed for CMOR 438 at Rice University.
"""

import numpy as np
from typing import Union, Sequence

ArrayLike = Union[np.ndarray, Sequence]


def _validate(y_true: ArrayLike, y_pred: ArrayLike):
    """Convert inputs to 1D float arrays and check they match in length."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {yt.shape[0]} and {yp.shape[0]}."
        )
    return yt, yp


# Regression metrics

def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Squared Error."""
    yt, yp = _validate(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    yt, yp = _validate(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of determination (R²).

    Returns 1.0 for a perfect fit. Returns 0.0 if the model
    performs no better than predicting the mean.
    """
    yt, yp = _validate(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# Classification metrics

def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Proportion of correctly classified samples."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(yt == yp))


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """
    Compute a confusion matrix.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
        Rows are true classes, columns are predicted classes.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([yt, yp]))
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp):
        cm[idx[t], idx[p]] += 1
    return cm