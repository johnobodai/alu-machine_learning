#!/usr/bin/env python3
"""A function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    This function takes a matrix as input and determines its shape, returning
    a list where each element corresponds to the size of a dimension.

    Parameters:
    - matrix (list): The input matrix for which the shape is to be determined.

    Returns:
    - list: A list representing the shape of the matrix, where each element
            corresponds to the size of a dimension.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if matrix:
            matrix = matrix[0]
        else:
            break
    return shape
