#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix"""


def minor(matrix):
    """Calculate the minor matrix of a matrix.

    Args:
        matrix (list): The input matrix for which the minor matrix is to be calculated.

    Returns:
        list: The minor matrix of the input matrix.

    Raises:
        TypeError: If the input is not a list of lists.
        ValueError: If the matrix is not square or is empty.

    Note:
        - The minor matrix of a given matrix is a matrix formed by the determinants of its submatrices.
    """
    # Check if input is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if the input matrix is empty
    if len(matrix) == 0 or not isinstance(matrix, list):
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if the matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Calculate the minor matrix
    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[0])):
            minor_row.append(determinant([row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]))
        minor_matrix.append(minor_row)

    return minor_matrix
