#!/usr/bin/env python3
"""Calculates the minor matrix of a matrix"""


def minor(matrix):
    """Calculates the minor matrix of a matrix.

    Args:
        matrix (list): Input matrix.

    Returns:
        list: Minor matrix of the input.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not a square or is empty.

    Note:
        Minor matrix: Formed by the determinants of submatrices.
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
            sub_matrix = [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]
            determinant_value = determinant(sub_matrix)
            minor_row.append(determinant_value)
        minor_matrix.append(minor_row)

    return minor_matrix
