#!/usr/bin/env python3
import numpy as np
""" Correlation Matrix from a covariance matrix"""


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.

    Args:
        C: A numpy.ndarray of shape (d, d) containing the covariance matrix.
            d is the number of dimensions.

    Returns:
        A numpy.ndarray of shape (d, d) containing the correlation matrix.

    Raises:
        TypeError: If C is not a 2D numpy.ndarray.
        ValueError: If C is not a square matrix.
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diag = np.diag(C).copy()

    # Ensure no division by zero with a small value
    diag[diag == 0] = 1e-10

    # Calculate standard deviations using the safe copy
    std_dev = np.sqrt(diag)

    # Normalize covariance matrix
    corr_matrix = C / np.outer(std_dev, std_dev)

    return corr_matrix
