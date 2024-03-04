#!/usr/bin/env python3
"""A function that calculates the determinant of a square matrix."""

def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    This function takes a square matrix as input and returns its determinant.

    Parameters:
    - matrix (list): The input square matrix for which the determinant is to be calculated.

    Returns:
    - float: The determinant of the input matrix.
    
    Raises:
    - TypeError: If matrix is not a list of lists.
    - ValueError: If matrix is not square.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    num_rows = len(matrix)
    num_cols = len(matrix[0]) if matrix else 0

    if num_rows != num_cols:
        raise ValueError("matrix must be a square matrix")

    # Base case for 0x0 matrix
    if num_rows == 0:
        return 1  # 0x0 matrix has determinant 1

    # Base case for 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # Recursive calculation of determinant using cofactor expansion along the first row
    det = 0
    for j in range(num_cols):
        sign = (-1) ** j
        submatrix = [row[:j] + row[j+1:] for row in matrix[1:]]  # Exclude jth column
        det += sign * matrix[0][j] * determinant(submatrix)

    return det
