#!/usr/bin/env python3
"""A function that calculates the shape of a numpy.ndarray."""


def np_shape(matrix):
    """
    Calculates the shape of a numpy.ndarray.

    Parameters:
    - matrix (numpy.ndarray): The input numpy array for which the
                              shape is to be determined.

    Returns:
    - tuple: A tuple representing the shape of the matrix.
    """
    shape = (*matrix.shape,)
    return shape
