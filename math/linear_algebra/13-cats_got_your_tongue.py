#!/usr/bin/env python3
"""A function that concatenates two matrices along a specific axis using NumPy."""

import numpy as np

def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis using NumPy.

    Parameters:
    - mat1 (numpy.ndarray): The first input numpy array.
    - mat2 (numpy.ndarray): The second input numpy array.
    - axis (int, optional): The axis along which the matrices will be concatenated. Default is 0.

    Returns:
    - numpy.ndarray: A new numpy array resulting from the concatenation.
    """
    return np.concatenate((mat1, mat2), axis=axis)

