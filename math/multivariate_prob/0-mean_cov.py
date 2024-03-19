#!/usr/bin/env python3
"""Calculating the mean and covariance"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data set.
            n is the number of data points, d is the number of dimensions.

    Returns:
        mean: A numpy.ndarray of shape (1, d)
        containing the mean of the data set.
        cov: A numpy.ndarray of shape (d, d) containing the covariance matrix.

    Raises:
        TypeError: If X is not a 2D numpy.ndarray.
        ValueError: If X contains less than 2 data points.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape

    # Calculate the mean
    mean = np.mean(X, axis=0, keepdims=True)

    # Center the data
    X_centered = X - mean

    # Calculate the covariance matrix (without using numpy.cov)
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
