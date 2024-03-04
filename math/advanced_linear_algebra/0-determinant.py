#!/usr/bin/env python3
"""A function that calculates the determinant of a matrix."""

def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    This function takes a matrix as input and returns its determinant.

    Parameters:
    - matrix (list): The input matrix for which the determinant is to be calculated.

    Returns:
    - float: The determinant of the input matrix.

    Raises:
    - TypeError: If matrix is not a list of lists.
    - ValueError: If matrix is not a square matrix.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)
    num_cols = len(matrix[0]) if matrix else 0

    if num_rows == 0 or num_cols != num_rows:
        if num_rows == 0:
            return 1  # 0x0 matrix has determinant 1
        raise ValueError("matrix must be a square matrix")

    if num_rows == 1:
        return matrix[0][0]

    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for x in range(num_rows):
        submatrix = [row[:x] + row[x+1:] for row in matrix[1:]]
        det += matrix[0][x] * determinant(submatrix) * (-1) ** x

    return det

