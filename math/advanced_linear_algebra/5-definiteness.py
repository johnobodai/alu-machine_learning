#!/usr/bin/env python3
""" Advanced Linear Algebra """

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a matrix.

    Args:
    - matrix (numpy.ndarray): Input matrix

    Returns:
    - str: Definiteness type ('Positive definite', 'Positive semi-definite',
           'Negative definite', 'Negative semi-definite', or 'Indefinite')
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eig(matrix)

    # Determine definiteness
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"

