#!/usr/bin/env python3
"""This module provides a function for calculating the determinant 
    of a matrix."""


def determinant(matrix):
    """Calculate the determinant of a matrix.

    Args:
        matrix (list): The input matrix for which the
                       determinant is to be calculated.

    Returns:
        int or float: The determinant of the input matrix.

    Raises:
        TypeError: If the input is not a list of lists
        or if the matrix is empty.
        ValueError: If the matrix is not square.

    Note:
        - The function handles the special case of a
        0x0 matrix by returning 1.
        - The function recursively calculates the
        determinant using the Laplace expansion.
    """
    # Check if input is a list of lists
    if not all(isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if the input matrix is empty
    if len(matrix) == 0 or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Handle the special case of a 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if the matrix is square
    if not all(len(r) == len(matrix) for r in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases for 1x1 and 2x2 matrices
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]) - \
            (matrix[0][1] * matrix[1][0])

    # Recursive calculation of determinant using Laplace expansion
    det_value = 0
    for idx, num in enumerate(matrix):
        submatrix = []
        pivot = matrix[0][idx]
        for row in matrix[1:]:
            new_row = []
            for j in range(len(matrix)):
                if j != idx:
                    new_row.append(row[j])
            submatrix.append(new_row)
        det_value += pivot * determinant(submatrix) * (-1) ** idx
    return det_value
