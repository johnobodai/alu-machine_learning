#!/usr/bin/env python3
"""A function that performs matrix multiplication using NumPy."""

import numpy as np

def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication using NumPy's dot product.

    Parameters:
    - mat1 (numpy.ndarray): The first input numpy array.
    - mat2 (numpy.ndarray): The second input numpy array.

    Returns:
    - numpy.ndarray: A new numpy array resulting from the matrix multiplication.
                    Returns None if the matrices cannot be multiplied.
    """
    return np.dot(mat1, mat2)
